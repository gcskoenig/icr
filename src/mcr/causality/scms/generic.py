from mcr.causality.scms import StructuralCausalModel
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyro
import torch
import jax.random as jrandom
import numpy as np
import logging
from mcr.backend.dist import MultivariateIndependent

from mcr.causality.scms._utils import numpyrodist_to_pyrodist

class StructuralFunction:

    def __init__(self, fnc, inv=None, additive=False):
        self.fnc = fnc
        self.inv_fnc = inv
        self.additive = additive

    def __call__(self, *args, **kwargs):
        return self.fnc(*args, **kwargs)

    def is_invertible(self):
        return (not self.inv_fnc is None) or (self.additive)

    def inv(self, x_pa, x_j, *args, **kwargs):
        if self.inv_fnc is None:
            if self.additive:
                if isinstance(x_pa, torch.Tensor):
                    zero = torch.tensor([0.0])
                else:
                    zero = jnp.array([0.0])
                x_j_wo = self.fnc(x_pa, zero)
                u_j = x_j - x_j_wo
                return u_j
            else:
                raise RuntimeError('Function is not inv')
        else:
            return self.inv_fnc(x_pa, x_j, *args, **kwargs)


class GenericSCM(StructuralCausalModel):

    SMALL_VAR = 0.0001

    def __init__(self, dag, fnc_dict={}, fnc_torch_dict={}, noise_dict={},
                 u_prefix='u_', sigmoidal=[]):
        super(GenericSCM, self).__init__(dag, u_prefix=u_prefix)

        for node in self.topological_order:
            if node not in fnc_dict:
                def fnc(x_pa, u_j):
                    result = u_j.flatten()
                    if x_pa.shape[1] > 0:
                        mean_pars = jnp.sum(x_pa, axis=1)
                        result = mean_pars.flatten() + result
                    return result
                fnc = StructuralFunction(fnc, additive=True)
                fnc_dict[node] = fnc
            if node not in fnc_torch_dict:
                def fnc(x_pa, u_j):
                    result = u_j.flatten()
                    if x_pa.shape[1] > 0:
                        mean_pars = torch.sum(x_pa, axis=1)
                        result = mean_pars.flatten() + result
                    return result
                fnc = StructuralFunction(fnc, additive=True)
                fnc_torch_dict[node] = fnc
            if node not in noise_dict:
                noise_dict[node] = dist.Normal(0, 0.1)
            # if node not in noise_torch_dict:
            #     noise_torch_dict[node] = pyro.distributions.Normal(0, 0.1)
            self.model[node]['sigmoidal'] = node in sigmoidal
            self.model[node]['fnc'] = fnc_dict[node]
            self.model[node]['fnc_torch'] = fnc_torch_dict[node]
            self.model[node]['noise_distribution'] = noise_dict[node]
            # self.model[node]['noise_distribution_torch'] = noise_torch_dict[node]

        self.fnc_dict = fnc_dict
        self.noise_dict = noise_dict
        self.fnc_torch_dict = fnc_torch_dict
        # self.noise_torch_dict = noise_torch_dict

    def save(self, filepath):
        raise NotImplementedError('Not implemented yet.')

    @staticmethod
    def load(filepath):
        raise NotImplementedError('Not implemented yet.')

    @staticmethod
    def _mcmc(num_samples, warmup_steps, num_chains, model, *args,
              rng_key=jrandom.PRNGKey(0), progress_bar=False, verbose=False, **kwargs):
        nuts_kernel = numpyro.infer.NUTS(model)
        mcmc = numpyro.infer.MCMC(
            nuts_kernel,
            num_samples=num_samples,
            num_warmup=warmup_steps,
            num_chains=num_chains,
            progress_bar=progress_bar
        )
        mcmc.run(rng_key, *args, **kwargs)
        if verbose:
            mcmc.print_summary()
        return mcmc

    @staticmethod
    def _mcmc_mixed(num_samples, warmup_steps, num_chains, model, *args,
                    rng_key=jrandom.PRNGKey(0), progress_bar=False, verbose=False, **kwargs):
        nuts_kernel = numpyro.infer.MixedHMC(numpyro.infer.HMC(model, trajectory_length=1.2))
        mcmc = numpyro.infer.MCMC(
            nuts_kernel,
            num_samples=num_samples,
            num_warmup=warmup_steps,
            num_chains=num_chains,
            progress_bar=progress_bar
        )
        mcmc.run(rng_key, *args, **kwargs)
        if verbose:
            mcmc.print_summary()
        return mcmc

    def predict_log_prob_obs(self, x_pre, y_name, y=1):
        raise NotImplementedError('Not implemented yet')

    def _get_parent_values(self, node):
        vals = self.get_values(var_names=self.model[node]['parents'])
        return vals

    def compute_node(self, node):
        par_values = self._get_parent_values(node).to_numpy()
        noise_values = self.get_noise_values()[[self.u_prefix+node]].to_numpy().flatten()
        vals = self.model[node]['fnc'](par_values, noise_values)
        vals = torch.tensor(np.array(vals.flatten()))
        return vals

    def _abduct_node_par_mcmc(self, node, obs, warmup_steps=200, nr_samples=400,
                              nr_chains=1, **kwargs):
        fnc = self.model[node]['fnc']

        obs_df = obs.to_frame().T
        x_pa = obs_df[list(self.model[node]['parents'])].to_numpy()
        x_j = obs_df[[node]].to_numpy()

        # if the function is invertible
        if fnc.is_invertible():
            u_j = fnc.inv(x_pa, x_j).flatten()
            return dist.Delta(v=u_j)

        # parent and node were observed
        def model(x_pa, x_j=None):
            u_j_dist = self.model[node]['noise_distribution']
            u_j = numpyro.sample("{}{}".format(self.u_prefix, node), u_j_dist)
            input = u_j
            if np.prod(x_pa.shape) > 0:
                # input = jnp.mean(x_pa).flatten() + u_j
                input = self.model[node]['fnc'](x_pa, u_j)
            x_j = numpyro.sample(node, dist.Normal(input, GenericSCM.SMALL_VAR), obs=x_j)

        mcmc_res = GenericSCM._mcmc(nr_samples, warmup_steps, nr_chains, model, x_pa,
                                    x_j=x_j, rng_key=jrandom.PRNGKey(0))

        type_dist = type(self.model[node]['noise_distribution'])
        smpl = mcmc_res.get_samples(False)["{}{}".format(self.u_prefix, node)]
        if type_dist is dist.Normal or type_dist is torch.distributions.Normal:
            return dist.Normal(smpl.mean(), smpl.std())
        elif type_dist is dist.Binomial or type_dist is torch.distributions.Binomial:
            return dist.Binomial(probs=smpl.mean())
        elif type_dist is dist.MixtureSameFamily:
            logging.debug("Using normal to approximate Mixture")
            return dist.Normal(smpl.mean(), smpl.std())
        else:
            raise NotImplementedError('distribution type not implemented.')

    def _abduct_node_par_unobs(self, node, obs, **kwargs):
        # one parent not observed, but parents of the parent were observed
        # ATTENTION: We assume that the noise variable was already abducted together with the unobserved parent's noise
        return None

    def _abduct_node_obs_discrete(self, node, obs, nr_samples=1000, temperature=1):
        # prepare data
        obs_df = obs.to_frame().T
        obs_df_dummy = obs_df.copy()
        obs_df_dummy[node] = np.array(0.0)
        x_pa = torch.tensor(obs_df[list(self.model[node]['parents'])].to_numpy())
        x_ch_dict = {}
        x_ch_pa_dict = {}
        for ch in self.model[node]['children']:
            x_ch_dict[ch] = torch.tensor(obs_df[[ch]].to_numpy().flatten())
            x_ch_pa_dict[ch] = torch.tensor(obs_df_dummy[list(self.model[ch]['parents'])].to_numpy())

        # pyro model for discrete inference of the binary target x_j
        def model_binary(x_pa, x_ch_dict=None, x_ch_pa_dict=None):
            input = torch.tensor(0.0)
            if torch.prod(torch.tensor(x_pa.shape)) > 0:
                # input = jnp.mean(x_pa).flatten() + u_j
                input = torch.sum(x_pa, axis=1)
            input = torch.sigmoid(input)
            x_j = pyro.sample(node, pyro.distributions.Bernoulli(probs=input), infer={"enumerate": "sequential"})
            x_chs = []
            chs = self.model[node]['children']
            with pyro.plate("child", len(chs)) as ch_ix:
                ch = chs[ch_ix]
                # TODO replace v-structure with child-node
                u_ch_dist_numpyro = self.model[ch]['noise_distribution']
                u_ch_dist = numpyrodist_to_pyrodist(u_ch_dist_numpyro)
                u_ch = pyro.sample("{}{}".format(self.u_prefix, ch), u_ch_dist)
                # insert x_j value into parent array
                ix = list(self.model[ch]['parents']).index(node)
                x_ch_pa = x_ch_pa_dict[ch]
                x_ch_pa[:, ix] = x_j
                x_ch_input = self.model[ch]['fnc_torch'](x_ch_pa, u_ch)
                x_ch = pyro.sample("x_{}".format(ch), pyro.distributions.Normal(x_ch_input, GenericSCM.SMALL_VAR),
                                   obs=x_ch_dict[ch])
                x_chs.append(x_ch)
            return x_j, x_chs

        xj_posterior_model = pyro.infer.infer_discrete(model_binary, first_available_dim=-1, temperature=temperature)
        samples = []
        for jj in range(nr_samples):
            samples.append(xj_posterior_model(x_pa, x_ch_dict=x_ch_dict, x_ch_pa_dict=x_ch_pa_dict)[0].numpy())

        prob = np.mean(samples)
        return dist.Binomial(probs=prob)


    def _abduct_node_obs_mcmc(self, node, obs, warmup_steps=1000, nr_samples=1000,
                              nr_chains=1, **kwargs):
        # if the node itself was not observed, but all other variables
        assert self.model[node]['sigmoidal']
        assert isinstance(self.model[node]['noise_distribution'], dist.Uniform)

        # infer discrete latent node distribution
        d_node = self._abduct_node_obs_discrete(node, obs, nr_samples=nr_samples)

        # prepare data
        obs_df = obs.to_frame().T
        obs_df_dummy = obs_df.copy()
        obs_df_dummy[node] = 0.0
        x_pa = obs_df[list(self.model[node]['parents'])].to_numpy()
        x_ch_dict = {}
        x_ch_pa_dict = {}
        for ch in self.model[node]['children']:
            x_ch_dict[ch] = jnp.array(obs_df[[ch]].to_numpy().flatten())
            x_ch_pa_dict[ch] = jnp.array(obs_df_dummy[list(self.model[ch]['parents'])].to_numpy())

        # numpyro model for the inference of the children distributions given the respective states
        def model_binary(x_j_obs, x_ch_dict=None, x_ch_pa_dict=None):
            x_j = numpyro.sample(node, d_node, obs=x_j_obs)
            for ch in self.model[node]['children']:
                u_ch_dist = self.model[ch]['noise_distribution']
                u_ch = numpyro.sample("{}{}".format(self.u_prefix, ch), u_ch_dist)
                # insert x_j value into parent array
                ix = list(self.model[ch]['parents']).index(node)
                x_ch_pa = x_ch_pa_dict[ch]
                x_ch_pa_mod = x_ch_pa.at[:, ix].set(x_j)
                x_ch_input = self.model[ch]['fnc'](x_ch_pa_mod, u_ch)
                x_ch = numpyro.sample("x_{}".format(ch), dist.Normal(x_ch_input, GenericSCM.SMALL_VAR), obs=x_ch_dict[ch])


        assert self.model[node]['sigmoidal']
        input = jnp.mean(x_pa, axis=1).flatten()[0]
        activation = 1 / (1 + jnp.exp(-input))


        x_j = 0
        mcmc_res = GenericSCM._mcmc(warmup_steps, nr_samples, nr_chains,
                                    model_binary, x_j, x_ch_dict=x_ch_dict, x_ch_pa_dict=x_ch_pa_dict)

        ds_0 = []
        unif_0 = dist.Uniform(activation, 1)
        ds_0.append(unif_0)
        for ch in self.model[node]['children']:
            smpl = mcmc_res.get_samples()[self.u_prefix+ch]
            if isinstance(self.model[ch]['noise_distribution'], dist.Normal):
                d = dist.Normal(np.mean(smpl), np.std(smpl))
                ds_0.append(d)
            else:
                raise NotImplementedError('only normal distibrution supported so far')

        x_j = 1
        mcmc_res = GenericSCM._mcmc(warmup_steps, nr_samples, nr_chains,
                                    model_binary, x_j, x_ch_dict=x_ch_dict, x_ch_pa_dict=x_ch_pa_dict)

        ds_1 = []
        unif_1 = dist.Uniform(0, activation)
        ds_1.append(unif_1)
        for ch in self.model[node]['children']:
            smpl = mcmc_res.get_samples()[self.u_prefix+ch]
            if isinstance(self.model[ch]['noise_distribution'], dist.Normal):
                d = dist.Normal(np.mean(smpl), np.std(smpl))
                ds_1.append(d)
            else:
                raise NotImplementedError('only normal distibrution supported so far')

        mv_d = MultivariateIndependent([ds_0, ds_1])
        mixing_dist = dist.Categorical(probs=np.array([1.0-d_node.probs, d_node.probs]))
        mixture = dist.MixtureSameFamily(mixing_dist, mv_d)

        return mixture

    # def predict_log_prob_individualized_obs(self, obs_pre, obs_post, intv_dict, y_name, y=1,
    #                                         nr_samples=1000, temperature=1):
    #     """Individualized post-recourse prediction, i.e. P(Y_post = y | x_pre, x_post)"""
    #
    #     # collect data from obs_pre and obs_post, covert into torch.tensors and make accesible in structured dicts
    #
    #     obs_df_pre = obs_pre.to_frame().T
    #     obs_df_post = obs_post.to_frame().T
    #
    #     obs_df_dummy_pre = obs_df_pre.copy()
    #     obs_df_dummy_post = obs_df_post.copy()
    #
    #     obs_df_dummy_pre[y_name] = np.array(0.0)
    #     obs_df_dummy_post[y_name] = np.array(0.0)
    #
    #     x_pa_pre = torch.tensor(obs_df_pre[list(self.model[y_name]['parents'])].to_numpy())
    #     x_pa_post = torch.tensor(obs_df_post[list(self.model[y_name]['parents'])].to_numpy())
    #
    #     x_ch_dict_pre = {}
    #     x_ch_dict_post = {}
    #
    #     x_ch_pa_dict_pre = {}
    #     x_ch_pa_dict_post = {}
    #
    #     for ch in self.model[y_name]['children']:
    #         x_ch_dict_pre[ch] = torch.tensor(obs_df_pre[[ch]].to_numpy().flatten())
    #         x_ch_dict_post[ch] = torch.tensor(obs_df_post[[ch]].to_numpy().flatten())
    #
    #         x_ch_pa_dict_pre[ch] = torch.tensor(obs_df_dummy_pre[list(self.model[ch]['parents'])].to_numpy())
    #         x_ch_pa_dict_post[ch] = torch.tensor(obs_df_dummy_post[list(self.model[ch]['parents'])].to_numpy())
    #
    #     # pyro model for discrete inference of the binary target y given x_pre and x_post
    #     def model_binary(x_pa, x_ch_dict=None, x_ch_pa_dict=None):
    #         input = torch.tensor(0.0)
    #         if torch.prod(torch.tensor(x_pa.shape)) > 0:
    #             # input = jnp.mean(x_pa).flatten() + u_j
    #             input = torch.sum(x_pa, axis=1)
    #         input = torch.sigmoid(input)
    #         x_j = pyro.sample(y_name, pyro.distributions.Bernoulli(probs=input), infer={"enumerate": "sequential"})
    #         x_chs = []
    #         for ch in self.model[y_name]['children']:
    #             u_ch_dist_numpyro = self.model[ch]['noise_distribution']
    #             u_ch_dist = numpyrodist_to_pyrodist(u_ch_dist_numpyro)
    #             u_ch = pyro.sample("{}{}".format(self.u_prefix, ch), u_ch_dist)
    #             # insert x_j value into parent array
    #             ix = list(self.model[ch]['parents']).index(y_name)
    #             x_ch_pa = x_ch_pa_dict[ch]
    #             x_ch_pa[:, ix] = x_j
    #             x_ch_input = self.model[ch]['fnc_torch'](x_ch_pa, u_ch)
    #             x_ch = pyro.sample("x_{}".format(ch), pyro.distributions.Normal(x_ch_input, GenericSCM.SMALL_VAR),
    #                                obs=x_ch_dict[ch])
    #             x_chs.append(x_ch)
    #         return x_j, x_chs
    #
    #     xj_posterior_model = pyro.infer.infer_discrete(model_binary, first_available_dim=-1, temperature=temperature)
    #     samples = []
    #     for jj in range(nr_samples):
    #         samples.append(xj_posterior_model(x_pa, x_ch_dict=x_ch_dict, x_ch_pa_dict=x_ch_pa_dict)[0].numpy())
    #
    #     prob = np.mean(samples)
    #     return dist.Binomial(probs=prob)
    #     raise NotImplementedError('Not implemented in abstract class.')
