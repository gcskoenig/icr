import numpyro.distributions as dist
from numpyro.distributions.util import is_prng_key
import jax.numpy as jnp
from functools import partial, cache
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


class BivariateBernoulli(dist.Distribution):

    def __init__(self, p1, p2, validate_args=None):
        self.p1 = jnp.array(p1)
        self.p2 = jnp.array(p2)
        assert self.p1.shape == self.p2.shape
        batch_shape = (1,)
        if len(self.p1.shape) > 0:
            batch_shape = (self.p1.shape[0],)
        event_shape = (2,)
        super(BivariateBernoulli, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        sample_shape_mod = list(sample_shape)
        sample_shape_mod[-1] = 1
        u = dist.Uniform(low=0.0, high=1.0).sample(key, tuple(sample_shape_mod))
        x1 = jnp.greater_equal(self.p1, u)
        x2 = jnp.greater_equal(self.p2, u)
        sample = jnp.squeeze(jnp.stack([x1, x2], axis=1), axis=-1)
        return sample

    def log_prob(self, values):
        """assuming value is one observation
        let (vl, vh) be an observation where pl <= ph, then
        p(0, 0) = 1 - ph
        p(1, 0) = 0
        p(0, 1) = ph - pl
        p(1, 1) = pl
        """
        pl, ph = self.p1, self.p2
        # if p1 >= p2 then flip array accordingly (such that 1 on left is less probable than on right)
        if self.p1 >= self.p2:
            pl, ph = self.p2, self.p1
            values = jnp.flip(values, axis=-1)

        # inner results given vl==1: pl if vh==1 and 0 if vh==0
        inner1 = jnp.where(values[..., 1], jnp.log(pl), jnp.log(0))
        # inner results given vl==0: ph-pl if vh==1 and 1-ph if vh==0
        inner0 = jnp.where(values[..., 1], jnp.log(ph - pl), jnp.log(1 - ph))
        # assigns inner result based on whether vl == 0 or ==1
        res = jnp.where(values[..., 0], inner1, inner0)
        return res


class BivariateInvertible(dist.Distribution):

    def __init__(self, d_j, fncs, xs_pa, y_ixs, validate_args=None):
        """
        fnc_pre/fnc_post are partial functions that have already been given the parent value as assigment
        """
        self.fnc_pre, self.fnc_post = fncs
        self.x_pa_pre, self.x_pa_post = xs_pa
        self.y_ix_pre, self.y_ix_post = y_ixs
        self.d_j = d_j
        batch_shape = (1,)
        if len(self.x_pa_pre.shape) > 0:
            batch_shape = (self.x_pa_pre.shape[0])
        super(BivariateInvertible, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=validate_args
        )

    def _get_completed_pa(self, ys):
        x_pa_pre = self.x_pa_pre.copy()
        if self.y_ix_pre is not None:
            x_pa_pre[..., self.y_ix_pre] = ys[0]
        x_pa_post = self.x_pa_post.copy()
        if self.y_ix_post is not None:
            x_pa_post[..., self.y_ix_post] = ys[1]
        return x_pa_pre, x_pa_post

    def _get_partial(self, ys):
        x_pa_pre, x_pa_post = self._get_completed_pa(ys)
        fnc_pre, fnc_post = partial(self.fnc_pre, x_pa_pre), partial(self.fnc_post, x_pa_post)
        return fnc_pre, fnc_post

    def _get_partial_inv(self, ys):
        x_pa_pre, x_pa_post = self._get_completed_pa(ys)
        fnc_pre_inv, fnc_post_inv = partial(self.fnc_pre.inv, x_pa_pre), partial(self.fnc_post.inv, x_pa_post)
        return fnc_pre_inv, fnc_post_inv

    def sample(self, key, sample_shape=(), ys=(1, 1)):
        assert is_prng_key(key)
        sample_shape_mod = list(sample_shape)
        sample_shape_mod[-1] = 1
        fnc_pre, fnc_post = self._get_partial(ys)
        u = self.d_j.sample(key, tuple(sample_shape_mod))
        x1 = fnc_pre(u)
        x2 = fnc_post(u)
        sample = jnp.squeeze(jnp.stack([x1, x2], axis=1), axis=-1)
        return sample

    def log_prob(self, value, ys=(1, 1)):
        fnc_pre_inv, fnc_post_inv = self._get_partial_inv(ys)
        u1 = fnc_pre_inv(value[..., 0])
        u2 = fnc_post_inv(value[..., 1])

        log_prob1 = self.d_j.log_prob(u1)
        u1_r = jnp.round(u1, decimals=2)
        d_2 = dist.Delta(u1_r)
        log_prob2 = d_2.log_prob(jnp.round(u2, decimals=2))
        # TODO make sure that both similar
        return log_prob1 + log_prob2


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
