import collections

from mcr.causality.scms import StructuralCausalModel
from mcr.distributions.utils import numpyrodist_to_pyrodist, add_uncertainty

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyro
import torch
import jax.random as jrandom
import numpy as np
import logging
from mcr.distributions.multivariate import MultivariateIndependent, BivariateBernoulli, BivariateInvertible, BivariateSigmoidal
from mcr.causality.scms.functions import linear_additive, linear_additive_torch
import collections


class GenericSCM(StructuralCausalModel):

    SMALL_VAR = 0.0001

    def __init__(self, dag, fnc_dict=None, fnc_torch_dict=None, noise_dict=None,
                 u_prefix='u_', sigmoidal=None, costs=None, y_name=None, bound_dict=None):
        super(GenericSCM, self).__init__(dag, u_prefix=u_prefix, costs=costs, y_name=y_name)

        if noise_dict is None:
            noise_dict = {}
        if fnc_torch_dict is None:
            fnc_torch_dict = {}
        if fnc_dict is None:
            fnc_dict = {}
        if sigmoidal is None:
            sigmoidal = []
        if bound_dict is None:
            bound_dict = {}

        for node in self.topological_order:
            if node not in fnc_dict:
                fnc_dict[node] = linear_additive
            if node not in fnc_torch_dict:
                fnc_torch_dict[node] = linear_additive_torch
            if node not in noise_dict:
                noise_dict[node] = dist.Normal(0, 0.1)
            if node not in bound_dict:
                bound_dict[node] = (float('-Inf'), float('Inf'))
                if node in sigmoidal:
                    bound_dict[node] = (0, 1)
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
        self.bounds = bound_dict
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

    def _get_parent_values(self, node):
        vals = self.get_values(var_names=self.model[node]['parents'])
        return vals

    def compute_node(self, node):
        if not self.model[node]['intervened']:
            par_values = self._get_parent_values(node).to_numpy()
            noise_values = self.get_noise_values()[[self.u_prefix+node]].to_numpy().flatten()
            vals = self.model[node]['fnc'](par_values, noise_values)
            vals = torch.tensor(np.array(vals.flatten()))
            return vals
        else:
            return super(GenericSCM, self).compute_node(node)

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
        elif fnc.is_transformable():
            d = fnc.transform(x_pa, x_j)
            return d

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
        elif type_dist is dist.Gamma or type_dist is torch.distributions.Gamma:
            raise NotImplementedError('Not implemented')
            # return dist.Normal(smpl.mean(), smpl.std())
        elif type_dist is dist.Uniform or type_dist is torch.distributions.Uniform:
            raise NotImplementedError('Not implemented.')
        elif type_dist is dist.Categorical or type_dist is torch.distributions.Categorical:
            return dist.Categorical(probs=np.array(smpl.value_counts()/len(smpl)))
        elif type_dist is dist.MixtureSameFamily:
            logging.debug("Using normal to approximate Mixture")
            return dist.Normal(smpl.mean(), smpl.std())
        else:
            raise NotImplementedError('distribution type not implemented.')

    def _abduct_node_par_unobs(self, node, obs, **kwargs):
        # one parent not observed, but parents of the parent were observed
        # ATTENTION: We assume that the noise variable was already abducted together with the unobserved parent's noise
        return None

    def _abduct_node_obs_discrete(self, node, obs):
        obs_df = obs.to_frame().T
        obs_df[node] = np.array(0.0)
        x_pa = obs_df[list(self.model[node]['parents'])].to_numpy()
        p_y = self.model[node]['fnc'].raw(x_pa)
        d_y = dist.Bernoulli(p_y.item())

        # then no further calculation needed
        if p_y == 1.0 or p_y == 0.0:
            return d_y

        def log_prob_joint(y):
            log_probs = []
            for ch in self.model[node]['children']:
                x_ch = obs_df[[ch]].to_numpy().flatten()
                x_ch_pa = obs_df[list(self.model[ch]['parents'])].to_numpy()
                d_u = self.model[ch]['noise_distribution']

                y_ix = list(self.model[ch]['parents']).index(node)
                x_ch_pa[..., y_ix] = y
                if self.model[node]['sigmoidal']:
                    p_ch_1 = self.model[node]['fnc'].raw(jnp.array([x_ch_pa]))
                    p_ch = x_ch * p_ch_1 + (1 - x_ch) * (1 - p_ch_1)
                    log_probs.append(jnp.log(p_ch))
                else:
                    u_ch = self.model[ch]['fnc'].inv(x_ch_pa, x_ch)
                    log_probs.append(d_u.log_prob(u_ch))
            log_probs.append(d_y.log_prob(y))
            return sum(log_probs)

        def log_posterior_y(y):
            nominator = log_prob_joint(y)
            denominator = jnp.log(jnp.exp(log_prob_joint(1)) + jnp.exp(log_prob_joint(0)))
            return (nominator - denominator).item()

        return dist.Binomial(probs=np.exp(log_posterior_y(1)), total_count=1)

    def _abduct_node_obs_discrete_pyro(self, node, obs, nr_samples=1000, temperature=1):
        # prepare data
        obs_df = obs.to_frame().T
        obs_df_dummy = obs_df.copy()
        obs_df_dummy[node] = np.array(0.0)
        x_pa = torch.tensor(obs_df[list(self.model[node]['parents'])].to_numpy(), dtype=torch.float64)
        x_ch_dict = {}
        x_ch_pa_dict = {}
        for ch in self.model[node]['children']:
            x_ch_dict[ch] = torch.tensor(obs_df[[ch]].to_numpy().flatten(), dtype=torch.float64)
            x_ch_pa_dict[ch] = torch.tensor(obs_df_dummy[list(self.model[ch]['parents'])].to_numpy(),
                                            dtype=torch.float64)

        # pyro model for discrete inference of the binary target x_j
        def model_binary(x_pa, x_ch_dict=None, x_ch_pa_dict=None):
            input = torch.tensor(0.0)
            if torch.prod(torch.tensor(x_pa.shape)) > 0:
                input = torch.sum(x_pa, dim=1)
            input = torch.sigmoid(input)
            x_j = pyro.sample(node, pyro.distributions.Bernoulli(probs=input), infer={"enumerate": "parallel"})
            x_chs = []
            chs = self.model[node]['children']
            for ch_ix in range(len(chs)):
            # with pyro.plate("child", len(chs)) as ch_ix:
                # get child latent distribution
                ch = chs[ch_ix]
                u_ch_dist_numpyro = self.model[ch]['noise_distribution']
                u_ch_dist = numpyrodist_to_pyrodist(u_ch_dist_numpyro)
                assert isinstance(u_ch_dist, pyro.distributions.Normal)

                # build parents array for child
                ix = list(self.model[ch]['parents']).index(node)
                x_ch_pa = x_ch_pa_dict[ch]
                if len(x_j.shape) > 0:
                    x_ch_pa_rep = x_ch_pa.repeat((len(x_j), 1))
                else:
                    x_ch_pa_rep = x_ch_pa
                x_ch_pa_rep[..., ix] = x_j

                # get function and assert additivity
                fnc = self.model[ch]['fnc_torch']
                assert fnc.additive

                # build respective target distribution
                x_ch_input = fnc.raw(x_ch_pa_rep).flatten()
                d_ch = pyro.distributions.Normal(x_ch_input + u_ch_dist.loc, u_ch_dist.scale)

                # define and append child
                x_ch = pyro.sample("x_{}".format(ch), d_ch, obs=x_ch_dict[ch].repeat(x_ch_input.shape))
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
        d_node = self._abduct_node_obs_discrete(node, obs)

        # prepare data
        obs_df = obs.to_frame().T
        obs_df_dummy = obs_df.copy()
        obs_df_dummy[node] = 0.0
        x_pa = obs_df[list(self.model[node]['parents'])].to_numpy()
        x_ch_dict = {}
        x_ch_pa_dict = {}
        chs_infer = []
        for ch in self.model[node]['children']:
            x_ch_dict[ch] = jnp.array(obs_df[[ch]].to_numpy().flatten())
            x_ch_pa_dict[ch] = jnp.array(obs_df_dummy[list(self.model[ch]['parents'])].to_numpy())
            if not self.model[ch]['fnc'].is_invertible() and not self.model[ch]['fnc'].is_transformable():
                chs_infer.append(ch)

        def complete_obs(node, x_pa_dummy, x_pa_val, x_pa_name):
            ix = list(self.model[node]['parents']).index(x_pa_name)
            x_pa_mod = x_pa_dummy.at[:, ix].set(x_pa_val)
            return x_pa_mod

        # numpyro model for the inference of the children distributions given the respective states
        def model_binary(x_j_obs, x_ch_dict=None, x_ch_pa_dict=None):
            x_j = numpyro.sample(node, d_node, obs=x_j_obs)
            for ch in chs_infer:
                u_ch_dist = self.model[ch]['noise_distribution']
                u_ch = numpyro.sample("{}{}".format(self.u_prefix, ch), u_ch_dist)
                # insert x_j value into parent array
                x_ch_pa_mod = complete_obs(ch, x_ch_pa_dict[ch], x_j, node)
                # ix = list(self.model[ch]['parents']).index(node)
                # x_ch_pa = x_ch_pa_dict[ch]
                # x_ch_pa_mod = x_ch_pa.at[:, ix].set(x_j)
                x_ch_input = self.model[ch]['fnc'](x_ch_pa_mod, u_ch)
                x_ch = numpyro.sample("x_{}".format(ch), dist.Normal(x_ch_input, GenericSCM.SMALL_VAR), obs=x_ch_dict[ch])

        assert self.model[node]['sigmoidal']
        activation = self.model[node]['fnc'].raw(x_pa).item()


        x_j = 0
        if len(chs_infer) > 0:
            mcmc_res = GenericSCM._mcmc(warmup_steps, nr_samples, nr_chains,
                                        model_binary, x_j, x_ch_dict=x_ch_dict, x_ch_pa_dict=x_ch_pa_dict)

        # TODO include latent distribution for discrete variables (in the form of the respective adapted uniform var)

        # TODO replace mcmc based sampling with analytical inference

        ds_0 = []
        unif_0 = dist.Uniform(activation, 1)
        ds_0.append(unif_0)
        for ch in self.model[node]['children']:
            if ch in chs_infer:
                smpl = mcmc_res.get_samples()[self.u_prefix+ch]
                if isinstance(self.model[ch]['noise_distribution'], dist.Normal):
                    d = dist.Normal(np.mean(smpl), np.std(smpl))
                    ds_0.append(d)
                else:
                    raise NotImplementedError('only normal distibrution supported so far')
            elif self.model[ch]['fnc'].is_invertible():
                x_pa_compl = complete_obs(ch, x_ch_pa_dict[ch], x_j, node)
                u_ch = self.model[ch]['fnc'].inv(x_pa_compl, x_j)
                d = dist.Delta(u_ch)
                ds_0.append(d)
            else:
                # assuming node is sigmoidal
                x_pa_compl = complete_obs(ch, x_ch_pa_dict[ch], x_j, node)
                d = self.model[ch]['fnc'].transform(x_pa_compl, x_ch_dict[ch])
                ds_0.append(d)



        x_j = 1
        if len(chs_infer):
            mcmc_res = GenericSCM._mcmc(warmup_steps, nr_samples, nr_chains,
                                        model_binary, x_j, x_ch_dict=x_ch_dict, x_ch_pa_dict=x_ch_pa_dict)

        ds_1 = []
        unif_1 = dist.Uniform(0, activation)
        ds_1.append(unif_1)
        for ch in self.model[node]['children']:
            if ch in chs_infer:
                smpl = mcmc_res.get_samples()[self.u_prefix+ch]
                if isinstance(self.model[ch]['noise_distribution'], dist.Normal):
                    d = dist.Normal(np.mean(smpl), np.std(smpl))
                    ds_1.append(d)
                else:
                    raise NotImplementedError('only normal distibrution supported so far')
            elif self.model[ch]['fnc'].is_invertible():
                x_pa_compl = complete_obs(ch, x_ch_pa_dict[ch], x_j, node)
                u_ch = self.model[ch]['fnc'].inv(x_pa_compl, x_j)
                d = dist.Delta(u_ch)
                ds_1.append(d)
            else:
                # assuming node is sigmoidal
                x_pa_compl = complete_obs(ch, x_ch_pa_dict[ch], x_j, node)
                d = self.model[ch]['fnc'].transform(x_pa_compl, x_ch_dict[ch])
                ds_1.append(d)

        mv_d = MultivariateIndependent([ds_0, ds_1])
        mixing_dist = dist.Categorical(probs=np.array([1.0-d_node.probs, d_node.probs]))
        mixture = dist.MixtureSameFamily(mixing_dist, mv_d)

        return mixture

    def predict_log_prob_obs(self, x_pre, y_name, y=1, **kwargs):
        """P(Y=y|X=x_pre)"""
        assert self.model[y_name]['fnc'].binary
        d = self._abduct_node_obs_discrete(y_name, x_pre)
        p = d.probs
        if len(p.shape) > 0:
            p = p.item()
        p = torch.tensor(p)
        return torch.log(p)

    def predict_log_prob_individualized_obs(self, obs_pre, obs_post, intv_dict, y_name, y=1):
        """analytical computation of the individualized post-recourse probability"""
        assert self.model[y_name]['sigmoidal']

        scm_post = self.do(intv_dict)
        p_y_pre = self.model[y_name]['fnc'].raw(jnp.array(obs_pre[list(self.model[y_name]['parents'])]))
        p_y_post = scm_post.model[y_name]['fnc'].raw(jnp.array(obs_post[list(scm_post.model[y_name]['parents'])]))

        # prepare data for the function input
        obs_df_dummy_pre = obs_pre.to_frame().T
        obs_df_dummy_post = obs_post.to_frame().T
        obs_df_dummy_pre[y_name] = np.array(0.0)
        obs_df_dummy_post[y_name] = np.array(0.0)

        # distribution that allows to compute the joint of pre/post observation of y
        dy = BivariateBernoulli(p_y_pre, p_y_post)

        # distribution that allows to compute the joint of pre/post observation of children

        # get distribution of the children with y as free parameter
        d_chs = {}
        val_chs = {}
        for ch in self.model[y_name]['children']:
            d_u_ch = self.model[ch]['noise_distribution']
            fnc_pre = self.model[ch]['fnc'] # structural functions
            fnc_post = scm_post.model[ch]['fnc']

            x_ch_pa_pre = obs_df_dummy_pre[list(self.model[ch]['parents'])].to_numpy()
            x_ch_pa_post = obs_df_dummy_post[list(scm_post.model[ch]['parents'])].to_numpy()
            y_ix_pre = list(self.model[ch]['parents']).index(y_name)
            y_ix_post = list(scm_post.model[ch]['parents']).index(y_name)

            if self.model[ch]['fnc'].is_invertible():
                d_ch = BivariateInvertible(d_u_ch, (fnc_pre, fnc_post), (x_ch_pa_pre, x_ch_pa_post), (y_ix_pre, y_ix_post))
            elif self.model[ch]['sigmoidal']:
                d_ch = BivariateSigmoidal(d_u_ch, (fnc_pre, fnc_post), (x_ch_pa_pre, x_ch_pa_post), (y_ix_pre, y_ix_post))
            else:
                raise NotImplementedError('only invertible structural equation or sigmoidal supported so far.')
            d_chs[ch] = d_ch

            val_chs[ch] = jnp.array([obs_pre[ch], obs_post[ch]])

        # TODO compile the components to a joint probability distribution over y and its markov blanket
        def log_joint_prob_ys(ys):
            """p(x, ys)"""
            ys = jnp.array(ys)
            log_probs_chs = [d_chs[ch].log_prob(val_chs[ch], ys=ys) for ch in d_chs.keys()]
            log_prob_ys = dy.log_prob(ys)
            res = sum(log_probs_chs) + log_prob_ys
            return res

        def marg_y_post(y_post):
            """p(x, y_post)"""
            ys_pre = [0, 1]
            probs_ys = [jnp.exp(log_joint_prob_ys([y_pre, y_post])) for y_pre in ys_pre]
            prob = sum(probs_ys)
            return prob

        def log_prob_y_post(y_post):
            """p(y_post|x)"""
            nominator = jnp.log(marg_y_post(y_post))
            denominator = jnp.log(marg_y_post(1) + marg_y_post(0))
            return nominator - denominator

        return log_prob_y_post(y)

    def predict_log_prob_individualized_obs_pyro(self, obs_pre, obs_post, intv_dict, y_name, y=1,
                                                 nr_samples=1000, temperature=1):
        """Individualized post-recourse prediction, i.e. P(Y_post = y | x_pre, x_post)"""

        # get post_intervention scm
        scm_post = self.do(intv_dict)

        # collect data from obs_pre and obs_post, covert into torch.tensors and make accesible in structured dicts

        obs_df_pre = obs_pre.to_frame().T
        obs_df_post = obs_post.to_frame().T

        obs_df_dummy_pre = obs_df_pre.copy()
        obs_df_dummy_post = obs_df_post.copy()

        obs_df_dummy_pre[y_name] = np.array(0.0)
        obs_df_dummy_post[y_name] = np.array(0.0)

        x_pa_pre = torch.tensor(obs_df_pre[list(self.model[y_name]['parents'])].to_numpy())
        x_pa_post = torch.tensor(obs_df_post[list(scm_post.model[y_name]['parents'])].to_numpy())

        x_ch_dict_pre = {}
        x_ch_dict_post = {}

        x_ch_pa_dict_pre = {}
        x_ch_pa_dict_post = {}

        for ch in self.model[y_name]['children']:
            x_ch_dict_pre[ch] = torch.tensor(obs_df_pre[[ch]].to_numpy().flatten())
            x_ch_pa_dict_pre[ch] = torch.tensor(obs_df_dummy_pre[list(self.model[ch]['parents'])].to_numpy())

        for ch in scm_post.model[y_name]['children']:
            x_ch_dict_post[ch] = torch.tensor(obs_df_post[[ch]].to_numpy().flatten())
            x_ch_pa_dict_post[ch] = torch.tensor(obs_df_dummy_post[list(scm_post.model[ch]['parents'])].to_numpy())

        # pyro model for discrete inference of the binary target y given x_pre and x_post

        def model_binary(x_pa_pre, x_pa_post, x_ch_dict_pre=None, x_ch_pa_dict_pre=None,
                         x_ch_dict_post=None, x_ch_pa_dict_post=None):
            assert not (x_pa_pre is None or x_pa_post is None or x_ch_pa_dict_pre is None or x_ch_pa_dict_post is None)

            # latent variable
            u_y_dist = numpyrodist_to_pyrodist(self.model[y_name]['noise_distribution'])
            u_y = pyro.sample('u_y', u_y_dist)

            # state of y
            y_p_pre = self.model[y_name]['fnc_torch'](x_pa_pre, u_y)
            y_p_pre = add_uncertainty(y_p_pre)
            y_pre = pyro.sample(y_name + '_pre', pyro.distributions.Bernoulli(probs=y_p_pre),
                                infer={"enumerate": "parallel"})

            y_p_post = scm_post.model[y_name]['fnc_torch'](x_pa_post, u_y)
            y_p_post = add_uncertainty(y_p_post)
            y_post = pyro.sample(y_name + '_post', pyro.distributions.Bernoulli(probs=y_p_post),
                                 infer={"enumerate": "parallel"})

            x_chs_pre = []

            u_chs = {}
            for ch in self.model[y_name]['children']:
                # sample noise variable
                u_ch_dist_numpyro = self.model[ch]['noise_distribution']
                u_ch_dist = numpyrodist_to_pyrodist(u_ch_dist_numpyro)
                u_ch = pyro.sample("{}{}".format(self.u_prefix, ch), u_ch_dist)

                u_chs[ch] = u_ch

                # insert x_j value into parent array
                ix = list(self.model[ch]['parents']).index(y_name)
                if len(y_pre.shape) > 0:
                    x_ch_pa = x_ch_pa_dict_pre[ch].repeat((len(y_pre), 1))
                else:
                    x_ch_pa = x_ch_pa_dict_pre[ch]
                x_ch_pa[..., ix] = y_pre

                # sample child
                x_ch_input = self.model[ch]['fnc_torch'](x_ch_pa, u_ch).flatten()
                x_ch = pyro.sample("x_{}_pre".format(ch), pyro.distributions.Normal(x_ch_input, GenericSCM.SMALL_VAR**2),
                                   obs=x_ch_dict_pre[ch].repeat(x_ch_input.shape))
                x_chs_pre.append(x_ch)

            x_chs_post = []
            for ch in scm_post.model[y_name]['children']:
                # get previously sampled latent state.
                u_ch = u_chs[ch]

                # insert x_j value into parent array
                ix = list(scm_post.model[ch]['parents']).index(y_name)
                if len(y_post.shape) > 0:
                    x_ch_pa = x_ch_pa_dict_post[ch].repeat((len(y_post), 1, 1))
                else:
                    x_ch_pa = x_ch_pa_dict_post[ch]
                x_ch_pa[..., ix] = y_post

                # sample child
                x_ch_input = scm_post.model[ch]['fnc_torch'](x_ch_pa, u_ch).flatten()
                x_ch = pyro.sample("x_{}_post".format(ch), pyro.distributions.Normal(x_ch_input, GenericSCM.SMALL_VAR**2),
                                   obs=x_ch_dict_post[ch].repeat(x_ch_input.shape))
                x_chs_post.append(x_ch)

            return y_pre, y_post, x_chs_pre, x_chs_post

        posterior_model = pyro.infer.infer_discrete(model_binary, first_available_dim=-1, temperature=temperature)
        samples = []
        for jj in range(nr_samples):
            samples.append(posterior_model(x_pa_pre, x_pa_post,
                                           x_ch_dict_pre=x_ch_dict_pre,
                                           x_ch_dict_post=x_ch_dict_post,
                                           x_ch_pa_dict_pre=x_ch_pa_dict_pre,
                                           x_ch_pa_dict_post=x_ch_pa_dict_post)[1].numpy())

        prob = np.mean(samples)
        return torch.log(torch.tensor(prob))
