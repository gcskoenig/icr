import logging
from jax import random
import numpy as np
from dataclasses import dataclass

from numpyro.distributions import constraints

import numpyro
import numpyro.distributions as dist
import time
from numpyro.infer import MCMC, NUTS

from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import ClippedAdam

numpyro.set_host_device_count(1)

logging.basicConfig(format="%(message)s", level=logging.INFO)
# numpyro.set_rng_seed(0)

# def sigmoidal_fnc(x_pa, u_j):
#     result = u_j
#     if x_pa.shape[0] > 0:
#         mean_pars = torch.mean(x_pa, axis=1)
#         result += mean_pars
#     result = torch.sigmoid(result)
#     return result.flatten()

@dataclass
class data:
    J = 1
    pa_y = np.array([-7, 3, 4])
    y = np.array([3])

@dataclass
class data2:
    J = 1
    pa_y = np.array([0])
    sp_y = np.array([2])
    ch_y = np.array([4])

def model_case1(pa_y, y):
    u_y = numpyro.sample("u_y", dist.Normal(0, 1))
    pa_y = pa_y.mean()
    y = numpyro.sample("y", dist.Normal(pa_y + u_y, 0.0001), obs=y)
    return y

def guide_case1(pa_y, y):
    mean_parameter = numpyro.param("mu", 0)
    std_parameter = numpyro.param("std", 1, constraint=constraints.positive)

    u_y = numpyro.sample("u_y", dist.Normal(mean_parameter, std_parameter))
    pa_y = pa_y.mean()
    y = numpyro.sample("y", dist.Normal(pa_y + u_y, 0.0001))
    return y

# def model(pa_y, sp_y):
#     u_y = numpyro.sample("u_y", dist.Normal(torch.zeros(data.J), torch.ones(data.J)))
#     y = pa_y + u_y
#     ch_y = numpyro.sample("ch_y", dist.Normal(y + sp_y, 0.3))
#     return y, ch_y
#
# def guide(pa_y, sp_y):
#     mean_parameter = numpyro.param("mu", torch.zeros(data.J))
#     std_parameter = numpyro.param("std", torch.ones(data.J), constraint=constraints.positive)
#
#     u_y = numpyro.sample("u_y", dist.Normal(mean_parameter, std_parameter))
#     y = pa_y + u_y
#     ch_y = numpyro.sample("ch_y", dist.Normal(y + sp_y, 0.3))
#     return y, ch_y

# def conditioned_model(model, forward_evidence, backward_evidence):
#     return poutine.condition(model, data=backward_evidence)(*forward_evidence)

def main_mcmc(num_samples, warmup_steps, num_chains, model, *args):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=warmup_steps,
        num_chains=num_chains,
    )
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, *args)
    mcmc.print_summary()
    return mcmc

def main_svi(optimization_steps, model, guide, *args):
    numpyro.clear_param_store()

    my_svi = SVI(model=model,
                 guide=guide,
                 optim=ClippedAdam({"lr": 0.01}),
                 loss=Trace_ELBO())

    start_time = time.time()

    for i in range(optimization_steps):

        loss = my_svi.step(*args)

        if (i % 100 == 0):
            print(f'iter: {i}, loss: {round(loss, 2)}', end="\r")

    end_time = time.time()
    print("Took {}s".format(end_time-start_time))
    print("mu: {}, std: {}".format(numpyro.param("mu"), numpyro.param("std")))
    return numpyro.param("mu"), numpyro.param("std")

#conditioned_m = numpyro.condition(model, data={'ch_y': data2.ch_y})

#conditioned_m_case1 = numpyro.condition(model_case1, data={'y': data.y})

main_mcmc(100, 500, 1, model_case1, data.pa_y, data.y)
# result = main_svi(5000, model_case1, guide_case1, data.pa_y, data.y)
