import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

import torch
import torch.distributions as tdist
from torch.distributions import constraints
import matplotlib.pyplot as plt

def scm():
    u1 = pyro.sample('u1', dist.Normal(0, 1))
    x1 = pyro.deterministic('x1', u1)
    uy = pyro.sample('uy', dist.Normal(0, 0.1))
    y = pyro.deterministic('y', x1 + uy)
    u2 = pyro.sample('u2', dist.Normal(0, 0.1))
    x2 = pyro.deterministic('x2', u2 + y)
    return x1, x2, y

scm_cond = pyro.condition(scm, data={'x1': torch.tensor(1.0), 'u1': torch.tensor(2.0)})

def guide():
    a = pyro.param('a', torch.tensor(1.0))
    b = pyro.param('b', torch.tensor(1.0))
    uy = pyro.sample('uy', dist.Normal(a, b))
    c = pyro.param('c', torch.tensor(1.0))
    d = pyro.param('d', torch.tensor(1.0))
    u1 = pyro.sample('u1', dist.Normal(c, d))
    y = pyro.deterministic('y', u1 + uy)
    return u1, uy, y

pyro.clear_param_store()
svi = pyro.infer.SVI(model=scm_cond,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 0.003}),
                     loss=pyro.infer.Trace_ELBO())


losses, a, b, c, d = [], [], [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step())
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())
    c.append(pyro.param("c").item())
    d.append(pyro.param("d").item())

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss");
plt.show()
print('a = ', pyro.param("a").item())
print('b = ', pyro.param("b").item())
print('c = ', pyro.param("c").item())
print('d = ', pyro.param("d").item())




def scm(parents):
    y = pyro.sample('y', dist.Normal(parents, 0.1))
    return y

def guide(parents):
    a = pyro.param('a', torch.tensor(1.0))
    b = pyro.param('b', torch.tensor(1.0), constraint=constraints.positive)
    y = pyro.sample('y', dist.Normal(parents + a, b))
    return y

scm_cond = pyro.condition(scm, data={'y': torch.tensor(3.0)})

pyro.clear_param_store()
svi = pyro.infer.SVI(model=scm_cond,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 0.003}),
                     loss=pyro.infer.Trace_ELBO())


losses = []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step(1.0))

print('a = ', pyro.param("a").item())
print('b = ', pyro.param("b").item())



eps = 0.00000001

def scm():
    u1 = pyro.sample('u1', dist.Normal(0, 1.0))
    uy = pyro.sample('uy', dist.Normal(0, 0.1))
    u2 = pyro.sample('u2', dist.Normal(0, 0.5))
    x1 = pyro.sample('x1', dist.Normal(u1, eps))
    y = pyro.sample('y', dist.Normal(uy + x1, eps))
    x2 = pyro.sample('x2', dist.Normal(u2 + y, eps))
    return x1, x2, y

def guide():
    a = pyro.param('a', torch.tensor(1.0))
    b = pyro.param('b', torch.tensor(1.0), constraint=constraints.positive)
    uy = pyro.sample('uy', dist.Normal(a, b))
    return uy

scm_cond = pyro.condition(scm, data={'x1': torch.tensor(-1.0),
                                     'x2': torch.tensor(1.3)})


pyro.clear_param_store()
svi = pyro.infer.SVI(model=scm_cond,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 0.003}),
                     loss=pyro.infer.Trace_ELBO())


losses = []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step())

print('a = ', pyro.param("a").item())
print('b = ', pyro.param("b").item())