import jax.nn

from mcr.causality.scms import GenericSCM
from mcr.causality.dags import DirectedAcyclicGraph
import numpy as np
import numpyro
import numpyro.distributions as dist
from mcr.causality.scms.functions import *
import sys

# # EXAMPLE 1 SCM
#
# sigma_high = torch.tensor(0.5)
# sigma_medium = torch.tensor(0.09)
# sigma_low = torch.tensor(0.05)
#
# SCM_EX1 = BinomialBinarySCM(
#     dag=DirectedAcyclicGraph(
#         adjacency_matrix=np.array([[0, 1, 0],
#                                    [0, 0, 1],
#                                    [0, 0, 0]]),
#         var_names=['vaccinated', 'covid-free', 'symptom-free']
#     ),
#     p_dict={'vaccinated': sigma_high,
#             'symptom-free': sigma_low, 'covid-free': sigma_medium}
# )
#
# costs = np.array([0.5, 0.1])
# y_name = 'covid-free'
# SCM_EX1.set_prediction_target(y_name)

# GENERIC SCMS for experiments

y_name = 'y'

## NOISE DISTRIBUTIONS

unif_dist = numpyro.distributions.Uniform(low=jnp.array(0.0), high=jnp.array(1.0))
normal_dist = numpyro.distributions.Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))
normal_dist_small_var = numpyro.distributions.Normal(loc=jnp.array(0.0),
                                                    scale=jnp.array(0.1))

mixing_dist = numpyro.distributions.Categorical(probs=jnp.ones(3)/3)
multinormal_dist = numpyro.distributions.Normal(loc=jnp.array([-4, 0, 4]), scale=jnp.ones([3]))
mog_dist = numpyro.distributions.MixtureSameFamily(mixing_dist, multinormal_dist)


## SCMS

SCM_3_VAR_CAUSAL = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 1],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'x3', 'y']
    ),
    noise_dict={'x1': normal_dist, 'x2': normal_dist, 'x3': normal_dist_small_var, 'y': unif_dist},
    fnc_dict={y_name: sigmoidal_binomial},
    fnc_torch_dict={y_name: sigmoidal_binomial_torch},
    sigmoidal=[y_name],
    costs=[1.0, 1.0, 1.0],
    y_name=y_name
)

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
    sigmoidal=[y_name],
    costs=[1.0, 1.0, 1.0],
    y_name=y_name
)


fn_2 = lambda x_1, u_2:  -1 + 3 * jax.nn.sigmoid(-2 * x_1[..., 0]) + u_2
fn_2 = StructuralFunction(fn_2, additive=True)

fn_2_torch = lambda x_1, u_2: -1 + 3 * torch.sigmoid(-2 * x_1[..., 0]) + u_2
fn_2_torch = StructuralFunction(fn_2_torch, additive=True)

# assuming x is ordered as (x1, x2)
fn_3 = lambda x, u_3: -0.05 * x[..., 0] + 0.25 * x[..., 1]**2 + u_3
fn_3 = StructuralFunction(fn_3, additive=True)

# assuming the parents are ordered as (x3, y, x4)
fn_5 = lambda x, u_5: x[..., 0] * 0.2 - x[..., 1] - 0.2 * x[..., 2] + u_5
fn_5 = StructuralFunction(fn_5, additive=True)

SCM_5_VAR_NONLINEAR = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'x3', 'y', 'x4', 'x5']
    ),
    noise_dict={'x1': normal_dist, 'x2': normal_dist_small_var, 'x3': normal_dist, 'x4': normal_dist,
                'x5': normal_dist_small_var, 'y': unif_dist},
    fnc_dict={'x2': fn_2, 'x3': fn_3, 'x5': fn_5, 'y': sigmoidal_binomial},
    fnc_torch_dict={'x2': fn_2_torch, 'x3': fn_3, 'x5': fn_5, 'y': sigmoidal_binomial_torch},
    sigmoidal=['y'],
    costs=[1.0, 1.0, 1.0, 1.0, 1.0],
    y_name='y'
)


# COVID EXAMPLE
def unif_transform(raw_value, observed):
    if observed:
        return numpyro.distributions.Uniform(low=0.0, high=raw_value.item())
    else:
        return numpyro.distributions.Uniform(low=raw_value.item(), high=1.0)

fn_covid_raw = lambda x: jax.nn.sigmoid(-(-3 + x[..., 0] - x[..., 1] - 2.5 * x[..., 2] + 0.2 * x[..., 3]**2))
fn_covid = lambda x, u: jnp.greater_equal(fn_covid_raw(x), u)
fn_covid_transf = lambda x, x_j: unif_transform(fn_covid_raw(x), x_j)
fn_covid = StructuralFunction(fn_covid, raw=fn_covid_raw, transform=fn_covid_transf, binary=True)

fn_flu_raw = lambda x: jax.nn.sigmoid(-3.5 - 6 * x[..., 0])
fn_flu = lambda x, u: jnp.greater_equal(fn_flu_raw(x), u)
fn_flu_transf = lambda x, x_j: unif_transform(fn_flu_raw(x), x_j)
fn_flu = StructuralFunction(fn_flu, raw=fn_flu_raw, transform=fn_flu_transf, binary=True)

fn_appetite_raw = lambda x: jax.nn.sigmoid(- 2 * x[..., 0])
fn_appetite = lambda x, u: jnp.greater_equal(fn_appetite_raw(x), u)
fn_appetite_transf = lambda x, x_j: unif_transform(fn_appetite_raw(x), x_j)
fn_appetite = StructuralFunction(fn_appetite, raw=fn_flu_raw, transform=fn_appetite_transf, binary=True)

fn_fever_raw = lambda x: jax.nn.sigmoid(+ 6 - 9 * x[..., 0])
fn_fever = lambda x, u: jnp.greater_equal(fn_fever_raw(x), u)
fn_fever_transf = lambda x, x_j: unif_transform(fn_fever_raw(x), x_j)
fn_fever = StructuralFunction(fn_fever, raw=fn_fever_raw, transform=fn_fever_transf, binary=True)

fn_fatigue_raw = lambda x: jax.nn.sigmoid(-1 + x[..., 0]**2 - 2 * x[..., 1])
fn_fatigue = lambda x, u: jnp.greater_equal(fn_fatigue_raw(x), u)
fn_fatigue_transf = lambda x, x_j: unif_transform(fn_fatigue_raw(x), x_j)
fn_fatigue = StructuralFunction(fn_fatigue, raw=fn_fatigue_raw, transform=fn_fatigue_transf, binary=True)

SCM_COVID = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0]]),
        var_names=['pop_density', 'flu_shot', 'covid_shots', 'bmi_diff', 'covid-free', 'appetite_loss',
                   'fever', 'fatigue']
    ),
    noise_dict={'pop_density': dist.Gamma(4, rate=4/3),
                'flu_shot': dist.Bernoulli(probs=0.39),
                'covid_shots': dist.Categorical(probs=np.array([0.24, 0.02, 0.15, 0.59])),
                'bmi-diff': dist.Normal(0, 1),
                'covid-free': unif_dist,
                'flu': unif_dist,
                'appetite_loss': unif_dist,
                'fever': unif_dist,
                'fatigue': unif_dist
                },
    fnc_dict={'covid-free': fn_covid, 'flu': fn_flu, 'appetite': fn_appetite, 'fever': fn_fever,
              'fatigue': fn_fatigue},
    y_name= 'covid-free',
    sigmoidal=['covid-free', 'flu', 'appetite', 'fever', 'fatigue'],
    costs=np.ones(7),
    bound_dict={'covid_shots': (0, 3), 'flu_shot': (0, 1), 'pop_density': (0, float('Inf'))}
)

#

# COVID EXAMPLE

fn_skilled_raw = lambda x: jax.nn.sigmoid((-10 + 3*x[..., 0] + 4*x[..., 1]))
fn_skilled = lambda x, u: jnp.greater_equal(fn_skilled_raw(x), u)
fn_skilled_transf = lambda x, x_j: unif_transform(fn_skilled_raw(x), x_j)
fn_skilled = StructuralFunction(fn_skilled, raw=fn_skilled_raw,
                                transform=fn_skilled_transf, binary=True)

fn_nr_commits_raw = lambda x: 10 * x[..., 0] * (1 + 100 * x[..., 1])
fn_nr_commits = lambda x, u: fn_nr_commits_raw(x) + u
fn_nr_commits = StructuralFunction(fn_nr_commits, raw=fn_nr_commits_raw,
                                   additive=True)

fn_nr_languages_raw = lambda x: jax.nn.sigmoid(10 * x[..., 0])
fn_nr_languages = lambda x, u: fn_nr_languages_raw(x) + u
fn_nr_languages = StructuralFunction(fn_nr_languages, raw=fn_nr_languages_raw,
                                     additive=True)

fn_nr_stars_raw = lambda x: 10 * x[..., 0]
fn_nr_stars = lambda x, u: fn_nr_stars_raw(x) + u
fn_nr_stars = StructuralFunction(fn_nr_stars, raw=fn_nr_stars_raw,
                                 additive=True)



SCM_PROGRAMMING = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 1, 1, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0]]),
        var_names=['years_experience', 'degree', 'senior-level_skill', 'nr_commits', 'nr_languages', 'nr_stars']
    ),
    noise_dict={'years_experience': dist.GammaPoisson(8, rate=8/3),
                'degree': dist.Categorical(probs=np.array([0.4, 0.2, 0.3, 0.1])),
                'senior-level_skill': unif_dist,
                'nr_commits': dist.GammaPoisson(40, rate=40/4),
                'nr_languages': dist.GammaPoisson(2, rate=2/4),
                'nr_stars': dist.GammaPoisson(5, rate=5/4)
                },
    fnc_dict={'senior-level_skill': fn_skilled, 'nr_commits': fn_nr_commits, 'nr_stars': fn_nr_stars, 'fever': fn_fever,
              'fatigue': fn_fatigue},
    y_name='senior-level_skill',
    sigmoidal=['senior-level_skill'],
    costs=[5.0, 5.0, 0.0001, 0.01, 0.1],
    bound_dict={'years_experience': (0, sys.maxsize), 'degree': (0, 3),
                'nr_commits': (0, sys.maxsize),
                'nr_languages': (0, sys.maxsize),
                'nr_stars': (0, sys.maxsize)}
)




#  OVERVIEW

scm_dict = {'3var-noncausal': SCM_3_VAR_NONCAUSAL, '3var-causal': SCM_3_VAR_CAUSAL,
            '5var-nonlinear': SCM_5_VAR_NONLINEAR, '7var-covid': SCM_COVID,
            '5var-skill': SCM_PROGRAMMING
            }