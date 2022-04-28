from mcr.causality.scm import BinomialBinarySCM, GenericSCM
from mcr.causality.dags import DirectedAcyclicGraph
import torch
import numpy as np
import numpyro
import jax.numpy as jnp
import jax

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

unif_dist = numpyro.distributions.Uniform(0, 1)
normal_dist = numpyro.distributions.Normal(0, 1)

mixing_dist = numpyro.distributions.Categorical(probs=jnp.ones(3)/3)
multinormal_dist = numpyro.distributions.Normal(loc=jnp.array([-4, 0, 4]), scale=jnp.ones([3]))
mog_dist = numpyro.distributions.MixtureSameFamily(mixing_dist, multinormal_dist)

## functional relationships

def sigmoidal(x_pa, u_j):
    input = jnp.mean(x_pa, axis=1).flatten()
    output = 1/(1 + jnp.exp(-input))
    return output

def sigmoidal_binomial(x_pa, u_j):
    input = jnp.mean(x_pa, axis=1).flatten()
    input = 1/(1 + jnp.exp(-input))
    output = jnp.greater_equal(input, u_j.flatten())
    return output

def nonlinear_additive(x_pa, u_j, coeffs=None):
    if coeffs is None:
        coeffs = jnp.ones(x_pa.shape[1])
    input = 0
    for jj in range(len(coeffs)):
        input = input + jnp.power(x_pa[:, jj], jj+1)
    output = input.flatten() + u_j.flatten()
    return output

## SCMs

SCM_3_VAR_CAUSAL = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 1],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'x3', 'y']
    ),
    noise_dict={'x1': mog_dist, 'x2': normal_dist, 'x3': normal_dist, 'y': unif_dist},
    fnc_dict={y_name: sigmoidal_binomial}
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
    noise_dict={'x1': mog_dist, 'x2': normal_dist, 'x3': normal_dist, 'y': unif_dist},
    fnc_dict={y_name: sigmoidal_binomial}
)
SCM_3_VAR_CAUSAL.set_prediction_target(y_name)