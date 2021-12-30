import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

import matplotlib.pyplot as plt

import torch
import torch.distributions.constraints as constraints


def scm():
    p_y = torch.tensor(0.3)
    p_x1 = torch.tensor(0.5)
    p_x2 = torch.tensor(0.5)
    p_x3 = torch.tensor(0.05)

    x1 = pyro.sample('x1', dist.Binomial(probs=p_x1))
    x2 = pyro.sample('x2', dist.Binomial(probs=p_x2))
    sum = torch.remainder(x1 + x2, 2)
    u_y = pyro.sample('u_y', dist.Binomial(probs=p_y))
    y = torch.remainder(sum + u_y, 2)
    p_x3 = torch.add(torch.mul(y, 1.0 - p_x3), torch.mul(1.0 - y, p_x3))
    x3 = pyro.sample('x3', dist.Binomial(probs=p_x3))


def guide():
    #p_x1 = torch.tensor(0.5)
    #p_x2 = torch.tensor(0.5)
    #x1 = pyro.sample('x1', dist.Binomial(probs=p_x1))
    #x2 = pyro.sample('x2', dist.Binomial(probs=p_x2))
    const = constraints.unit_interval
    p_y = pyro.param('p_y', torch.tensor(0.3), constraint=const)
    u_y = pyro.sample('u_y', dist.Binomial(probs=p_y))


scm_cond = pyro.condition(scm, data={'x1': torch.tensor(0.0),
                                     'x2': torch.tensor(0.0),
                                     'x3': torch.tensor(1.0)})

pyro.clear_param_store()
svi = pyro.infer.SVI(model=scm_cond,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 0.01}),
                     loss=pyro.infer.Trace_ELBO())

losses, p_y= [], []
num_steps = 10000
thresh = 0.01
var_a = 0.1

jj = 0
while (jj < num_steps) and (var_a > thresh):
    jj += 1
    losses.append(svi.step())
    p_y.append(pyro.param("p_y").item())

print('p_y = ', pyro.param("p_y").item())

plt.figure()
plt.subplot(211)
plt.scatter(range(len(losses)), losses)
plt.subplot(212)
plt.scatter(range(len(losses)), p_y)
plt.show()
