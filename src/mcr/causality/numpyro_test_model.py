
import jax.numpy as jnp
import numpy as np

import numpyro

def model_fixed(y_obs=None, rng_key=None):
    x = 1.0 #jnp.array(1.0, dtype=jnp.float16)
    u_y_dist = numpyro.distributions.Normal(0, 1)
    u_y = numpyro.sample('u_y', u_y_dist, rng_key=rng_key)

    y_dist = numpyro.distributions.Normal(u_y + x, scale=0.0001)
    y = numpyro.sample('y', y_dist, obs=y_obs, rng_key=rng_key)
    return x, u_y, y