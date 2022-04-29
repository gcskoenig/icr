from mcr.causality.scms import StructuralCausalModel
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import torch
import jax.random as jrandom
import numpy as np
import logging
from mcr.backend.dist import MultivariateIndependent

from numpyro.infer import NUTS, MCMC, MixedHMC, HMC


class GenericSCM(StructuralCausalModel):

    SMALL_VAR = 0.0001

    def __init__(self, dag, fnc_dict={}, noise_dict={}, u_prefix='u_', binary=[]):
        super(GenericSCM, self).__init__(dag, u_prefix=u_prefix)

        for node in self.topological_order:
            if node not in fnc_dict:
                def fnc(x_pa, u_j):
                    #assert isinstance(x_pa, pd.DataFrame) and isinstance(u_j, pd.DataFrame)
                    result = u_j.flatten()
                    if x_pa.shape[1] > 0:
                        mean_pars = jnp.sum(x_pa, axis=1)
                        result = mean_pars.flatten() + result
                    return result
                fnc_dict[node] = fnc
            if node not in noise_dict:
                noise_dict[node] = dist.Normal(0, 0.1)
            self.model[node]['binary'] = node in binary
            self.model[node]['fnc'] = fnc_dict[node]
            self.model[node]['noise_distribution'] = noise_dict[node]

        self.fnc_dict = fnc_dict
        self.noise_dict = noise_dict

    def save(self, filepath):
        raise NotImplementedError('Not implemented yet.')

    @staticmethod
    def load(filepath):
        raise NotImplementedError('Not implemented yet.')

    @staticmethod
    def _mcmc(num_samples, warmup_steps, num_chains, model, *args, progress_bar=False, verbose=False, **kwargs):
        nuts_kernel = NUTS(model)
        rng_key = jrandom.PRNGKey(0)
        mcmc = MCMC(
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
    def _mcmc_mixed(num_samples, warmup_steps, num_chains, model, *args, progress_bar=False, verbose=False, **kwargs):
        nuts_kernel = MixedHMC(HMC(model, trajectory_length=1.2))
        rng_key = jrandom.PRNGKey(0)
        mcmc = MCMC(
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

    def _abduct_node_par_mcmc(self, node, obs, warmup_steps=200, nr_samples=800,
                              nr_chains=1, **kwargs):
        # parent and node were observed
        def model(x_pa, x_j=None):
            u_j_dist = self.model[node]['noise_distribution']
            u_j = numpyro.sample("{}{}".format(self.u_prefix, node), u_j_dist)
            input = u_j
            if np.prod(x_pa.shape) > 0:
                # input = jnp.mean(x_pa).flatten() + u_j
                input = self.model[node]['fnc'](x_pa, u_j)
            x_j = numpyro.sample(node, dist.Normal(input, GenericSCM.SMALL_VAR), obs=x_j)

        obs_df = obs.to_frame().T
        x_pa = obs_df[list(self.model[node]['parents'])].to_numpy()
        x_j = obs_df[[node]].to_numpy()
        mcmc_res = GenericSCM._mcmc(nr_samples, warmup_steps, nr_chains, model, x_pa, x_j=x_j)

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
        return numpyro.distributions.Delta()

    # def _abduct_node_obs_mcmc(self, node, obs, warmup_steps=3000, nr_samples=1000,
    #                           nr_chains=5, verbose=False, **kwargs):
    #     # if the node itself was not observed, but all other variables
    #     def model(x_pa, x_ch_dict=None, x_ch_pa_dict=None):
    #         u_j_dist = self.model[node]['noise_distribution']
    #         u_j = numpyro.sample("{}{}".format(self.u_prefix, node), u_j_dist)
    #         input = u_j
    #         if np.prod(x_pa.shape) > 0:
    #             # input = jnp.mean(x_pa).flatten() + u_j
    #             input = self.model[node]['fnc'](x_pa, u_j)
    #         x_j = input
    #         for ch in self.model[node]['children']:
    #             u_ch_dist = self.model[ch]['noise_distribution']
    #             u_ch = numpyro.sample("{}{}".format(self.u_prefix, ch), u_ch_dist)
    #             # insert x_j value into parent array
    #             ix = list(self.model[ch]['parents']).index(node)
    #             x_ch_pa = x_ch_pa_dict[ch]
    #             x_ch_pa_mod = x_ch_pa.at[:, ix].set(x_j)
    #             x_ch_input = self.model[ch]['fnc'](x_ch_pa_mod, u_ch)
    #             x_ch = numpyro.sample("x_{}".format(ch), dist.Normal(x_ch_input, GenericSCM.SMALL_VAR), obs=x_ch_dict[ch])
    #
    #     obs_df = obs.to_frame().T
    #     obs_df_dummy = obs_df.copy()
    #     obs_df_dummy[node] = 0.0
    #     x_pa = obs_df[list(self.model[node]['parents'])].to_numpy()
    #     x_ch_dict = {}
    #     x_ch_pa_dict = {}
    #     for ch in self.model[node]['children']:
    #         x_ch_dict[ch] = jnp.array(obs_df[[ch]].to_numpy().flatten())
    #         x_ch_pa_dict[ch] = jnp.array(obs_df_dummy[list(self.model[ch]['parents'])].to_numpy())
    #
    #     mcmc_res = GenericSCM._mcmc(warmup_steps, nr_samples, nr_chains, model, x_pa,
    #                                 x_ch_dict=x_ch_dict, x_ch_pa_dict=x_ch_pa_dict)
    #
    #     smpl = mcmc_res.get_samples(True)
    #     arr = np.array([smpl[key].flatten() for key in smpl.keys()]).T
    #
    #     if isinstance(self.model[node]['noise_distribution'], numpyro.distributions.Normal):
    #         mv_dist = numpyro.distributions.MultivariateNormal(loc=np.mean(arr, axis=0), covariance_matrix=np.cov(arr.T))
    #         return mv_dist
    #     # elif isinstance(self.model[node]['noise_distribution'], numpyro.distributions.Uniform):
    #     #     dss = []
    #     #     for jj in range(smpl[list(smpl.keys())[0]].shape[0]):
    #     #         ds = []
    #     #         elemns = list(self.model[node]['children'])
    #     #         elemns.insert(0, node)
    #     #         for ch in elemns:
    #     #             arr = smpl[self.u_prefix + ch][jj, :]
    #     #             node_dist = self.model[ch]['noise_distribution']
    #     #             if isinstance(node_dist, numpyro.distributions.Normal):
    #     #                 d = numpyro.distributions.Normal(np.mean(arr), np.std(arr))
    #     #             elif isinstance(node_dist, numpyro.distributions.Uniform):
    #     #                 d = numpyro.distributions.Uniform(np.quantile(arr, 0.05), np.quantile(arr, 0.95))
    #     #             else:
    #     #                 raise NotImplementedError('distribution type not implemented')
    #     #             ds.append(d)
    #     #         dss.append(ds)
    #     #
    #     #     mixing_dist = dist.Categorical(probs=jnp.ones(len(dss)) / len(dss))
    #     #     mv_d = MultivariateIndependent(dss)
    #     #     mixture = dist.MixtureSameFamily(mixing_dist, mv_d)
    #     #     return mixture
    #     else:
    #         raise NotImplementedError('only multivariate normal supported so far')

    def _abduct_node_obs_mcmc(self, node, obs, warmup_steps=3000, nr_samples=1000,
                              nr_chains=5, **kwargs):
        assert self.model[node]['binary']
        # if the node itself was not observed, but all other variables

        def model_binary(x_pa, x_ch_dict=None, x_ch_pa_dict=None):
            input = [0.0]
            if np.prod(x_pa.shape) > 0:
                # input = jnp.mean(x_pa).flatten() + u_j
                input = jnp.sum(x_pa, axis=1)
            input = 1/(1 + jnp.exp(-input))
            x_j = numpyro.sample(node, numpyro.distributions.Bernoulli(probs=input[0]))
            for ch in self.model[node]['children']:
                u_ch_dist = self.model[ch]['noise_distribution']
                u_ch = numpyro.sample("{}{}".format(self.u_prefix, ch), u_ch_dist)
                # insert x_j value into parent array
                ix = list(self.model[ch]['parents']).index(node)
                x_ch_pa = x_ch_pa_dict[ch]
                x_ch_pa_mod = x_ch_pa.at[:, ix].set(x_j)
                x_ch_input = self.model[ch]['fnc'](x_ch_pa_mod, u_ch)
                x_ch = numpyro.sample("x_{}".format(ch), dist.Normal(x_ch_input, GenericSCM.SMALL_VAR), obs=x_ch_dict[ch])

        obs_df = obs.to_frame().T
        obs_df_dummy = obs_df.copy()
        obs_df_dummy[node] = 0.0
        x_pa = obs_df[list(self.model[node]['parents'])].to_numpy()
        x_ch_dict = {}
        x_ch_pa_dict = {}
        for ch in self.model[node]['children']:
            x_ch_dict[ch] = jnp.array(obs_df[[ch]].to_numpy().flatten())
            x_ch_pa_dict[ch] = jnp.array(obs_df_dummy[list(self.model[ch]['parents'])].to_numpy())

        mcmc_res = GenericSCM._mcmc_mixed(warmup_steps, nr_samples, nr_chains, model_binary, x_pa,
                                          x_ch_dict=x_ch_dict, x_ch_pa_dict=x_ch_pa_dict)

        smpl = mcmc_res.get_samples(True)
        arr = np.array([smpl[key].flatten() for key in smpl.keys()]).T

        if isinstance(self.model[node]['noise_distribution'], numpyro.distributions.Uniform):
            dss = []
            for jj in range(smpl[list(smpl.keys())[0]].shape[0]):
                ds = []
                elemns = list(self.model[node]['children'])
                elemns.insert(0, node)
                for ch in elemns:
                    arr = smpl[self.u_prefix + ch][jj, :]
                    node_dist = self.model[ch]['noise_distribution']
                    if isinstance(node_dist, numpyro.distributions.Normal):
                        d = numpyro.distributions.Normal(np.mean(arr), np.std(arr))
                    elif isinstance(node_dist, numpyro.distributions.Uniform):
                        d = numpyro.distributions.Uniform(np.quantile(arr, 0.05), np.quantile(arr, 0.95))
                    else:
                        raise NotImplementedError('distribution type not implemented')
                    ds.append(d)
                dss.append(ds)

            mixing_dist = dist.Categorical(probs=jnp.ones(len(dss)) / len(dss))
            mv_d = MultivariateIndependent(dss)
            mixture = dist.MixtureSameFamily(mixing_dist, mv_d)
            return mixture
        else:
            raise NotImplementedError('only multivariate normal supported so far')



# class GenericSCM(StructuralCausalModel):
#
#     SMALL_VAR = 0.0001
#
#     def __init__(self, dag, fnc_dict={}, noise_dict={}, u_prefix='u_'):
#         super(GenericSCM, self).__init__(dag, u_prefix=u_prefix)
#
#         for node in self.topological_order:
#             if node not in fnc_dict:
#                 def fnc(x_pa, u_j):
#                     assert type(x_pa) == torch.Tensor and type(u_j) == torch.Tensor
#                     if x_pa.shape[1] > 0:
#                         mean_pars = torch.mean(x_pa, axis=1)
#                         result = mean_pars + u_j
#                         return result
#                     else:
#                         return u_j
#                 fnc_dict[node] = fnc
#             if node not in noise_dict:
#                 noise_dict[node] = dist.Normal(0, 1)
#             self.model[node]['fnc'] = fnc_dict[node]
#             self.model[node]['noise_distribution'] = noise_dict[node]
#
#         self.fnc_dict = fnc_dict
#         self.noise_dict = noise_dict
#
#     def save(self, filepath):
#         raise NotImplementedError('Not implemented yet.')
#         # self.dag.save(filepath)
#         # fnc_dict = {}
#         # noise_dict = {}
#         # for node in self.dag.var_names:
#         #     fnc_dict[node] = self.model[node]['fnc']
#         #     noise_dict[node] = self.model[node]['noise_distribution']
#         # scm_dict = {}
#         # scm_dict['fnc_dict'] = fnc_dict
#         # scm_dict['noise_dict'] = noise_dict
#         # scm_dict['y_name'] = self.predict_target
#         #
#         # try:
#         #     with open(filepath + '_scm_dict.json', 'w') as f:
#         #         json.dump(scm_dict, f)
#         # except Exception as exc:
#         #     logging.warning('Could not save scm_dict.json')
#         #     logging.info('Exception: {}'.format(exc))
#
#     @staticmethod
#     def _mcmc(num_samples, warmup_steps, num_chains, model, *args):
#         nuts_kernel = NUTS(model, jit_compile=False)
#         mcmc = MCMC(
#             nuts_kernel,
#             num_samples=num_samples,
#             warmup_steps=warmup_steps,
#             num_chains=num_chains,
#         )
#         mcmc.run(*args)
#         # mcmc.summary(prob=0.5)
#         return mcmc
#
#     @staticmethod
#     def _svi(optimization_steps, model, guide, *args):
#         pyro.clear_param_store()
#         my_svi = SVI(model=model,
#                      guide=guide,
#                      optim=ClippedAdam({"lr": 0.001}),
#                      loss=Trace_ELBO())
#
#         for i in range(optimization_steps):
#             loss = my_svi.step(*args)
#             if (i % 100 == 0):
#                 print(f'iter: {i}, loss: {round(loss, 2)}', end="\r")
#
#     @staticmethod
#     def load(filepath):
#         raise NotImplementedError('Not implemented yet.')
#
#     def predict_log_prob_obs(self, x_pre, y_name, y=1):
#         raise NotImplementedError('Not implemented yet')
#
#     def _get_parent_values(self, node):
#         vals = []
#         for par in self.model[node]['parents']:
#             vals.append(self.model[par]['values'])
#         if len(vals) > 0:
#             x_pa = torch.stack(vals, dim=1)
#             return x_pa
#         else:
#             return torch.empty(0)
#
#     def compute_node(self, node):
#         par_values = self._get_parent_values(node)
#         noise_values = self.model[node]['noise_values']
#         vals = self.model[node]['fnc'](par_values, noise_values)
#         self.model[node]['values'] = vals.flatten()
#         return self.model[node]['values']
#
#     def _abduct_node_par_mcmc(self, node, obs, warmup_steps=100, nr_samples=200,
#                               nr_chains=1, **kwargs):
#         # parent and node were observed
#         def model(x_pa):
#             u_j_dist = self.model[node]['noise_distribution']
#             u_j = pyro.sample("u_j", u_j_dist).reshape(1,)
#             x_j = self.model[node]['fnc'](x_pa, u_j)
#             x_j = pyro.sample("x_j", dist.Normal(x_j, GenericSCM.SMALL_VAR))
#
#         model_c = pyro.condition(model, data={node: obs[node]})
#         x_pa = torch.tensor(obs.to_frame().T[list(self.model[node]['parents'])].to_numpy())
#         mcmc_res = GenericSCM._mcmc(warmup_steps, nr_samples, nr_chains, model_c, x_pa)
#
#
#         type_dist = type(self.model[node]['noise_distribution'])
#         smpl = mcmc_res.get_samples(5000)['u_j']
#         if type_dist is dist.Normal or type_dist is torch.distributions.Normal:
#             return dist.Normal(smpl.mean(), smpl.std())
#         elif type_dist is dist.Binomial or type_dist is torch.distributions.Binomial:
#             return dist.Binomial(probs=smpl.mean())
#
#     def _abduct_node_par_svi(self, node, obs, svi_steps=10**3, **kwargs):
#         # parent and node were observed
#         def model(x_pa):
#             u_j_dist = self.model[node]['noise_distribution']
#             u_j = pyro.sample("u_j", u_j_dist).reshape(1,)
#             x_j = self.model[node]['fnc'](x_pa, u_j)
#             x_j = pyro.sample("x_j", dist.Normal(x_j, GenericSCM.SMALL_VAR))
#
#         type_dist = type(self.model[node]['noise_distribution'])
#
#         def guide(x_pa):
#             if type_dist is dist.Normal or type_dist is torch.distributions.Normal:
#                 mu = pyro.param("mu", torch.zeros(1))
#                 std = pyro.param("std", torch.ones(1), constraint=constraints.positive)
#                 u_j = pyro.sample('u_j', dist.Normal(mu, std))
#             elif type_dist is dist.Binomial or type_dist is torch.distributions.Binomial:
#                 probs = pyro.param("probs", torch.tensor(0.5), constraint=constraints.interval(0, 1))
#                 u_j = pyro.sample('u_j', dist.Binomial(probs=probs))
#             else:
#                 raise NotImplementedError(type_dist + ' not supported.')
#             x_j = self.model[node]['fnc'](x_pa, u_j)
#             x_j = pyro.sample("x_j", dist.Normal(x_j, GenericSCM.SMALL_VAR))
#
#         model_c = pyro.condition(model, data={node: obs[node]})
#         x_pa = torch.tensor(obs.to_frame().T[list(self.model[node]['parents'])].to_numpy())
#         GenericSCM._svi(svi_steps, model_c, guide, x_pa)
#
#         # create result distribution
#         if type_dist is dist.Normal or type_dist is torch.distributions.Normal:
#             return dist.Normal(pyro.param('mu'), pyro.param('std'))
#         elif type_dist is dist.Binomial or type_dist is torch.distributions.Binomial:
#             return dist.Binomial(probs=pyro.param('probs'))
#
#     def _abduct_node_par_unobs(self, node, obs, scm_abd, **kwargs):
#         # one parent not observed, but parents of the parent were observed
#         raise NotImplementedError('Not implemented yet')
#
#     def _abduct_node_obs(self, node, obs, **kwargs):
#         # if the node itself was not observed, but all other variables
#         raise NotImplementedError('Not implemented yet')

