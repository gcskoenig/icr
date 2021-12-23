import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF



import pandas as pd
import numpy as np
import torch
from torch.distributions import constraints

from rfi.backend.causality.sem import RandomGPGaussianNoiseSEM
from rfi.backend.causality.dags import DirectedAcyclicGraph
from rfi.examples import SyntheticExample

sigma_medium = 0.5
sigma_low = 0.1

ii_audit = SyntheticExample(
    name='ii-model-audit',
    sem=RandomGPGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 1, 1],
                                       [0, 0, 0],
                                       [0, 0, 0]]),
            var_names=['x1', 'x2', 'y']
        ),
        noise_std_dict={'x1': sigma_medium, 'x2': sigma_medium, 'y': sigma_low}
    )
)


import seaborn as sns
import matplotlib.pyplot as plt


# ii_audit.sem._random_function('x2', torch.distributions.Normal(0, 1).sample((1000,)).flatten())
#
#
# values = torch.distributions.Normal(0, 1).sample((1000,1))
#
# seed = np.random.randint(2**16)
#
gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), optimizer=None)
# node_values = torch.tensor(gp.sample_y(values, random_state=seed)).reshape(-1)
#
# plt.scatter(values, node_values)
# plt.show()

def scm():
    x1 = pyro.sample('x1', dist.Normal(0.0, 1.0))
    tmp = torch.tensor(gp.sample_y(x1)).reshape(-1)
    x2 = pyro.sample('x2', dist.Normal(tmp, 0.1))
    return x1, x2

def guide():
    a = pyro.param('a', torch.tensor(0.0))
    b = pyro.param('b', torch.tensor(1.0), constraint=constraints.positive)
    x1 = pyro.sample('x1', dist.Normal(a, b))
    return x1

scm_cond = pyro.condition(scm, data={'x2': torch.tensor(-.7)})


pyro.clear_param_store()
svi = pyro.infer.SVI(model=scm_cond,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 0.01}),
                     loss=pyro.infer.Trace_ELBO())


losses, a, b = [], [], []
num_steps = 10000s
thresh=0.01
var_a = 0.1

jj = 0
while (jj < num_steps) and (var_a > thresh):
    jj += 1
    losses.append(svi.step())
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())

print('a = ', pyro.param("a").item())
print('b = ', pyro.param("b").item())

plt.subplot(311)
plt.scatter(range(len(losses)), losses)
plt.subplot(312)
plt.scatter(range(len(losses)), a)
plt.subplot(313)
plt.scatter(range(len(losses)), b)
plt.show()


def model_to_df(model, names, size, *args, **kwargs):
    data = []
    for _ in range(size):
        data.append(model(*args, **kwargs))

    data = np.array(data)
    df = pd.DataFrame(data, columns=names)
    return df

names = ['x1', 'x2']
size = 10**3
df_cond = model_to_df(scm_cond, names, size)
df_uncond = model_to_df(scm, names, size)

scm_cond2 = pyro.condition(scm, data={'x2': torch.tensor(-1.6)})
df_cond2 = model_to_df(scm_cond2, names, size)

sns.pairplot(df_cond)
plt.show()
sns.pairplot(df_cond2)
plt.show()
sns.pairplot(df_uncond)
plt.show()


import pyro.contrib.gp as gp
from gpytorch.models import ApproximateGP

variational_strategy = None
gp = ApproximateGP(variational_strategy)

model = gp.model()

kernel = gp.kernels.RBF(input_dim=1)
kernel.variance = torch.tensor(1.0)
kernel.lengthscale = torch.tensor(1.0)
gpr = gp.models.GPRegression(torch.tensor(), torch.tensor())