import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

from torch.distributions import constraints

import torch

import pandas as pd
import numpy as np


def model_to_df(model, names, size, *args, **kwargs):
    data = []
    for _ in range(size):
        data.append(model(*args, **kwargs))

    data = np.array(data)
    df = pd.DataFrame(data, columns=names)
    return df


def scm():
    u_1 = pyro.sample('u_1', dist.Normal(0, 0.1))
    x_1 = pyro.deterministic('x_1', u_1)
    u_y = pyro.sample('u_y', dist.Normal(0, 0.01))
    y = pyro.deterministic('y', x_1 + u_y)
    u_2 = pyro.sample('u_2', dist.Normal(0, 0.1))
    x_2 = pyro.deterministic('x_2', u_2 + y)
    return x_1, x_2, y


names = ['x1', 'x2', 'y']
df = model_to_df(scm, names, 1000)

scm_int0 = pyro.do(scm, data={'y': torch.tensor(0.0)})
scm_cond0 = pyro.condition(scm, data={'y': torch.tensor(0.0)})

posterior = pyro.infer.Importance(scm_cond0, num_samples=1000)

scm_int1 = pyro.do(scm, data={'y': torch.tensor(1.0)})
scm_cond1 = pyro.condition(scm, data={'y': torch.tensor(1.0)})

scm_int2 = pyro.do(scm, data={'y': torch.tensor(2.0)})
scm_cond2 = pyro.condition(scm, data={'y': torch.tensor(2.0)})


df_cond0 = model_to_df(scm_cond0, names, 1000)
df_cond1 = model_to_df(scm_cond1, names, 1000)
df_cond2 = model_to_df(scm_cond2, names, 1000)
df_cond = df_cond1.append(df_cond2)
df_cond = df_cond.append(df_cond0)

df_int0 = model_to_df(scm_int0, names, 1000)
df_int1 = model_to_df(scm_int1, names, 1000)
df_int2 = model_to_df(scm_int2, names, 1000)
df_int = df_int1.append(df_int2)
df_int = df_int.append(df_int0)

df_int.corr()
df_cond.corr()


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df_int)
plt.show()
sns.pairplot(df_cond)
plt.show()
sns.pairplot(df)
plt.show()

def parm_guide():
    a = pyro.param('a', torch.tensor(0.0))
    b = pyro.param('b', torch.tensor(1.0), constraint=constraints.positive)
    return pyro.sample('x2', dist.Normal(a, b))






###

def fnc(parent):
    u_i = pyro.sample('u_i', dist.Normal(0, 1))
    x_i = pyro.deterministic('x_i', parent + u_i)
    return x_i

def guide(parent):
    a = pyro.param('a', torch.tensor(1.0))
    b = pyro.param('b', torch.tensor(1.0), constraint=constraints.positive)
    return pyro.sample('u_i', dist.Normal(a*parent, b))


x_i_obs = torch.tensor(0.0)
fnc_c = pyro.condition(fnc, data={'x_i': x_i_obs})

pyro.clear_param_store()
svi = pyro.infer.SVI(model=fnc_c,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 0.003}),
                     loss=pyro.infer.Trace_ELBO())

parent = 3.0

losses, a, b = [], [], []
num_steps = 3000
for t in range(num_steps):
    losses.append(svi.step(parent))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");
print('a = ',pyro.param("a").item())
print('b = ', pyro.param("b").item())
plt.show()

plt.subplot(1,2,1)
plt.plot(a)
plt.ylabel('a')

plt.subplot(1,2,2)
plt.ylabel('b')
plt.plot(b)
plt.tight_layout()
plt.show()