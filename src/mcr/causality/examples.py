from mcr.causality.scms import BinomialBinarySCM, GenericSCM, StructuralFunction
from mcr.causality.dags import DirectedAcyclicGraph
import torch
import numpy as np
import numpyro
import jax.numpy as jnp
from mcr.causality.scms.functions import *

# EXAMPLE 1 SCM

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.05)

SCM_EX1 = BinomialBinarySCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 0]]),
        var_names=['vaccinated', 'covid-free', 'symptom-free']
    ),
    p_dict={'vaccinated': sigma_high,
            'symptom-free': sigma_low, 'covid-free': sigma_medium}
)

costs = np.array([0.5, 0.1])
y_name = 'covid-free'
SCM_EX1.set_prediction_target(y_name)

# GENERIC SCMS for experiments

y_name = 'y'

## noise distributions

unif_dist = numpyro.distributions.Uniform(low=jnp.array(0.0), high=jnp.array(1.0))
normal_dist = numpyro.distributions.Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))

mixing_dist = numpyro.distributions.Categorical(probs=jnp.ones(3)/3)
multinormal_dist = numpyro.distributions.Normal(loc=jnp.array([-4, 0, 4]), scale=jnp.ones([3]))
mog_dist = numpyro.distributions.MixtureSameFamily(mixing_dist, multinormal_dist)


## SCMs

SCM_3_VAR_CAUSAL = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 1],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'x3', 'y']
    ),
    noise_dict={'x1': normal_dist, 'x2': normal_dist, 'x3': normal_dist, 'y': unif_dist},
    fnc_dict={y_name: sigmoidal_binomial},
    fnc_torch_dict={y_name: sigmoidal_binomial_torch},
    sigmoidal=['y']
)

SCM_3_VAR_CAUSAL.set_prediction_target(y_name)

SCM_3_VAR_NONCAUSAL = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 1],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'y', 'x3']
    ),
    noise_dict={'x1': normal_dist, 'x2': normal_dist, 'x3': normal_dist, 'y': unif_dist},
    fnc_dict={y_name: sigmoidal_binomial},
    fnc_torch_dict={y_name: sigmoidal_binomial_torch},
    sigmoidal=['y']
)

SCM_3_VAR_CAUSAL.set_prediction_target(y_name)