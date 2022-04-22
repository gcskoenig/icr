import logging
from dataclasses import dataclass

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import time
from pyro.infer import MCMC, NUTS

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

logging.basicConfig(format="%(message)s", level=logging.INFO)
pyro.set_rng_seed(0)

@dataclass
class data:
    J = 1
    pa_y = torch.tensor([-7]).type(torch.Tensor)
    y = torch.tensor([-3]).type(torch.Tensor)

@dataclass
class data2:
    J = 1
    pa_y = torch.tensor([0]).type(torch.Tensor)
    sp_y = torch.tensor([2]).type(torch.Tensor)
    ch_y = torch.tensor([4]).type(torch.Tensor)

def model_case1(pa_y):
    u_y = pyro.sample("u_y", dist.Normal(torch.zeros(data.J), torch.ones(data.J)))
    y = pyro.sample("y", dist.Normal(pa_y + u_y, 0.0001))
    return y

def model(pa_y, sp_y):
    u_y = pyro.sample("u_y", dist.Normal(torch.zeros(data.J), torch.ones(data.J)))
    y = pa_y + u_y
    ch_y = pyro.sample("ch_y", dist.Normal(y + sp_y, 0.3))
    return y, ch_y

def guide(pa_y, sp_y):
    mean_parameter = pyro.param("mu", torch.zeros(data.J))
    std_parameter = pyro.param("std", torch.ones(data.J), constraint=constraints.positive)

    u_y = pyro.sample("u_y", dist.Normal(mean_parameter, std_parameter))
    y = pa_y + u_y
    ch_y = pyro.sample("ch_y", dist.Normal(y + sp_y, 0.3))
    return y, ch_y

# def conditioned_model(model, forward_evidence, backward_evidence):
#     return poutine.condition(model, data=backward_evidence)(*forward_evidence)

def main_mcmc(num_samples, warmup_steps, num_chains, model, *args):
    nuts_kernel = NUTS(model, jit_compile=False)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )
    mcmc.run(*args)
    mcmc.summary(prob=0.5)
    return mcmc

def main_svi(optimization_steps, model, guide):
    pyro.clear_param_store()

    my_svi = SVI(model=model,
                 guide=guide,
                 optim=ClippedAdam({"lr": 0.001}),
                 loss=Trace_ELBO())

    start_time = time.time()

    for i in range(optimization_steps):

        loss = my_svi.step(data2.pa_y, data2.sp_y)

        if (i % 100 == 0):
            print(f'iter: {i}, loss: {round(loss, 2)}', end="\r")

    end_time = time.time()
    print("Took {}s".format(end_time-start_time))
    print("mu: {}, std: {}".format(pyro.param("mu"), pyro.param("std")))
    return pyro.param("mu"), pyro.param("std")

conditioned_m = poutine.condition(model, data={'ch_y': data2.ch_y})
conditioned_g = poutine.condition(guide, data={'ch_y': data2.ch_y})

conditioned_m_case1 = pyro.condition(model_case1, data={'y': data.y})

main_mcmc(1000, 500, 1, conditioned_m, data2.pa_y, data2.sp_y)
result = main_svi(5000, conditioned_m, guide)
#mcmc = main_mcmc(1000, 1000, 1, conditioned_m_case1, data.pa_y)