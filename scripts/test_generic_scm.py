import logging
import mcr.causality.examples as ex
import numpy as np

logging.getLogger().setLevel(logging.INFO)

scm = ex.SCM_3_VAR_NONCAUSAL
y_name = 'y'
scm.set_prediction_target(y_name)

context = scm.sample_context(1000)
data = scm.compute()
print(1/(1+np.exp(-np.sum(data.iloc[0, :2]))))
print(context.iloc[0, 3])

scm_abd = scm.abduct(data.iloc[0, [0, 1, 3]], infer_type='mcmc')
cntxt = scm_abd.sample_context(1000)
smpl = scm_abd.compute()

import numpyro
import jax

pk = jax.random.PRNGKey(42)

def model():
    d = numpyro.distributions.Bernoulli(probs=0.5)
    x = numpyro.sample('x', d, rng_key=pk)
    return x

samples = []
for ii in range(100):
    samples.append(model())



for ii in range(10):
    print(context.iloc[ii])
    print(data.iloc[ii, :])
    scm.abduct(data.iloc[ii, [0, 1, 3]], infer_type='mcmc')


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# for jj in range(smpl[list(smpl.keys())[0]].shape[0]):
#     arr = np.array([smpl[key][jj, :] for key in smpl.keys()]).T
#
#     df = pd.DataFrame(arr, columns=smpl.keys())
#
#     sns.displot(df, x=df.columns[0], y=df.columns[1])
#     plt.show()
#     sns.displot(df['y'])
#     plt.show()
#     sns.displot(df['u_x3'])
#     plt.show()


arr = np.array([smpl[key].flatten() for key in smpl.keys()]).T
df = pd.DataFrame(arr, columns=smpl.keys())

sns.displot(df, x=df.columns[0], y=df.columns[1])
plt.show()
sns.displot(df['y'])
plt.show()
sns.displot(df['u_x3'])
plt.show()


import numpyro.distributions as dist
from mcr.backend.dist import MultivariateIndependent

import jax
import jax.numpy as jnp
import numpyro
import numpy as np

jk = jax.random.PRNGKey(42)

d1 = dist.Normal(0, 1)
d2 = dist.Uniform(0, 1)

dss = [[d2, d1, d2], [d1, d2, d2]]

mixing_dist = dist.Categorical(probs=jnp.ones(2)/2)
mv_d = MultivariateIndependent(dss)
mixture = dist.MixtureSameFamily(mixing_dist, mv_d)

smpl = mixture.sample(jk, sample_shape=(300,))
mixture.log_prob(smpl[10])

x1 = np.random.randn(1000)
x2 = np.random.randn(1000) + x1
x3 = np.random.randn(1000) + x2
arr = np.stack([x1, x2, x3])
mn = np.mean(arr, axis=1)
cov = np.cov(arr)

mns = np.stack([mn, mn+3])
covs = np.stack([cov, cov])


mixing_dist = dist.Categorical(probs=jnp.ones(2)/2)
component_dist = dist.MultivariateNormal(loc=mns, covariance_matrix=covs)
mixture = dist.MixtureSameFamily(mixing_dist, component_dist)

mixture.sample(jk, sample_shape=(500,))