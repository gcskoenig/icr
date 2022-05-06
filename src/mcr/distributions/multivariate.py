import numpyro.distributions as dist
from numpyro.distributions.util import is_prng_key
import jax.numpy as jnp
import torch

class MultivariateIndependent(dist.Distribution):

    def __init__(self, dss, validate_args=None):
        self.dss = dss
        for ds in dss:
            for d in ds:
                assert d.batch_shape == ()
                assert d.event_shape == ()
        batch_shape = (len(dss),)
        event_shape = (len(dss[0]),)
        super(MultivariateIndependent, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        sampless = []
        for ds in self.dss:
            samples = []
            for d in ds:
                s = d.sample(key, sample_shape)
                samples.append(s)
            samples = jnp.stack(samples, axis=1)
            sampless.append(samples)
        sampless = jnp.stack(sampless, axis=2)
        result = jnp.swapaxes(sampless, 1, 2)
        return result

    def log_prob(self, value):
        if len(value.shape) == 1:
            return jnp.sum([self.dss[0][i].log_prob(value[i]) for i in range(len(self.dss[0]))])
        else:
            lps = []
            for j in range(value.shape[0]):
                arr = jnp.array([self.dss[j][i].log_prob(value[j][i]) for i in range(len(self.dss[j]))])
                lp = jnp.sum(arr)
                lps.append(lp)
            return jnp.stack(lps)


class TransformedUniform(torch.distributions.Distribution):

    def __init__(self, sigma, p_y_1, **kwargs):
        self.phi = torch.tensor(sigma)
        self.p_y_1 = torch.tensor(p_y_1)

        self.p_smaller = self.p_y_1 / self.phi
        self.p_larger = (1 - self.p_y_1) / (1 - self.phi)

        self.factor_smaller = torch.tensor(1.0) / self.p_smaller
        self.factor_larger = torch.tensor(1.0) / self.p_larger

        super().__init__(**kwargs)

    def rsample(self, sample_shape=torch.Size()):
        v = torch.distributions.Uniform(0, 1).rsample(sample_shape)
        smpl = torch.min(v, self.p_y_1) * self.factor_smaller + torch.max(torch.tensor(0), v - self.p_y_1) * self.factor_larger
        return smpl

    def log_prob(self, value):
        res = (value <= self.phi) * self.p_smaller + (value > self.phi) * self.p_larger
        res = torch.log(res)
        return res