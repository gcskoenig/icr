import numpyro.distributions
import pandas as pd

from mcr.causality.scm import GenericSCM
from mcr.causality.dags import DirectedAcyclicGraph
import os
import numpy as np
import logging
import torch
import jax
import jax.numpy as jnp
import numpyro
import mcr.causality.examples as ex

logging.getLogger().setLevel(logging.INFO)

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.05)
sigma_verylow = torch.tensor(0.001)

def sigmoidal_fnc(x_pa, u_j):
    result = u_j
    if x_pa.shape[0] > 0:
        mean_pars = jnp.mean(x_pa, axis=1)
        result = u_j + mean_pars
    result = 1/(1 + jnp.exp(-result))
    return result.flatten()

def sum_fnc(x_pa, u_j):
    result = u_j.flatten()
    if x_pa.shape[0] > 0:
        sum_pars = jnp.sum(x_pa, axis=1).flatten()
        result = u_j + sum_pars
    return result.flatten()

binomial_noise = numpyro.distributions.Binomial(probs=jnp.array([0.5]))

x = numpyro.sample('x', binomial_noise, rng_key=jax.random.PRNGKey(10))

scm = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 0, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]]),
        var_names=['vaccinated', 'covid-free', 'symptom-free', 'entrance', 'assigned-building']
    ),
    #fnc_dict={'covid-free': sum_fnc, 'vaccinated': sum_fnc, 'symptom-free': sum_fnc, 'entrance': sum_fnc,
    #          'assigned-building': sum_fnc},
    noise_dict={'vaccinated': numpyro.distributions.Normal(0, 1)}
                #'vaccinated': binomial_noise, 'symptom-free': binomial_noise,
    #            'entrance': binomial_noise, 'assigned-building': binomial_noise}
)

costs = np.array([0.5, 0.1, 0.1, 0.1])
y_name = 'covid-free'
scm.set_prediction_target(y_name)

# scm = ex.SCM_3_VAR_CAUSAL

context = scm.sample_context(1000)
data = scm.compute()

# import seaborn as sns
# from mcr.causality.scm import GenericSCM
# from mcr.causality.dags import DirectedAcyclicGraph
# import os
# import numpy as np
# import logging
# import torch
# import jax.numpy as jnp
# import mcr.causality.examples as ex
#
# logging.getLogger().setLevel(logging.INFO)
#
# sigma_high = torch.tensor(0.5)
# sigma_medium = torch.tensor(0.09)
# sigma_low = torch.tensor(0.05)
# sigma_verylow = torch.tensor(0.001)
#
# def sigmoidal_fnc(x_pa, u_j):
#     result = u_j
#     if x_pa.shape[0] > 0:
#         mean_pars = jnp.mean(x_pa, axis=1)
#         result = u_j + mean_pars
#     result = 1/(1 + jnp.exp(-result))
#     return result.flatten()
#
# scm = GenericSCM(
#     dag=DirectedAcyclicGraph(
#         adjacency_matrix=np.array([[0, 1, 0, 0, 0],
#                                    [0, 0, 1, 0, 0],
#                                    [0, 0, 0, 0, 0],
#                                    [0, 0, 0, 0, 1],
#                                    [0, 0, 0, 0, 0]]),
#         var_names=['vaccinated', 'covid-free', 'symptom-free', 'entrance', 'assigned-building']
#     )#,
#     #fnc_dict={'covid-free': sigmoidal_fnc}
# )
#
# costs = np.array([0.5, 0.1, 0.1, 0.1])
# y_name = 'covid-free'
# scm.set_prediction_target(y_name)
#
# # scm = ex.SCM_3_VAR_CAUSAL
#
# context = scm.sample_context(1000)
# data = scm.compute()
#
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# # sns.pairplot(data)
# # plt.show()
#
# scm_abd = scm.abduct_node('covid-free', data.iloc[0, [0, 2, 3, 4]], infer_type='mcmc')
# # # scm_abd_svi = scm.abduct_node('covid-free', data.iloc[0, :], infer_type='svi')
# print(context.iloc[0, :])
# scm_abd.mean
# import matplotlib.pyplot as plt
#
# sns.pairplot(data)
# plt.show()

print(context.iloc[0, :])
print(data.iloc[0, :])
scm_abd = scm.abduct(data.iloc[0, [0, 2, 3, 4]], infer_type='mcmc')
context_abd = scm_abd.sample_context(1000)
scm_abd.compute().describe()