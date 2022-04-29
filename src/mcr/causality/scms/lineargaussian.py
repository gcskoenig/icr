from mcr.causality.scms.scm import StructuralCausalModel
from mcr.causality.dags import DirectedAcyclicGraph
import torch
import numpy as np
from torch.distributions import Normal
from mcr.estimation.gaussian_estimator import GaussianConditionalEstimator


class LinearGaussianNoiseSCM(StructuralCausalModel):

    def __init__(self, dag: DirectedAcyclicGraph,
                 coeff_dict={},
                 noise_std_dict={},
                 default_coeff=(0.0, 1.0),
                 default_noise_std_bounds=(0.0, 1.0),
                 seed=None,
                 u_prefix='u_'):
        """
        Args:
            dag: DAG, defining SEM
            coeff_dict: Coefficients of linear combination of parent nodes, if not written - considered as zero
            noise_std_dict: Noise std dict for each variable
            default_noise_std_bounds: Default noise std, if not specified in noise_std_dict.
                Sampled from U(default_noise_std_bounds[0], default_noise_std_bounds[1])
        """
        super(LinearGaussianNoiseSCM, self).__init__(dag, u_prefix=u_prefix)

        self.INVERTIBLE = True

        np.random.seed(seed) if seed is not None else None

        for node in self.topological_order:
            self.model[node]['noise_distribution'] = Normal(0, noise_std_dict[node]) if node in noise_std_dict \
                else Normal(0, np.random.uniform(*default_noise_std_bounds))

            coeff = None

            if node in coeff_dict:
                coeff = coeff_dict[node]
            else:
                if type(default_coeff) == float:
                    coeff = {par: default_coeff for par in self.model[node]['parents']}
                else:
                    coeff = {par: np.random.uniform(*default_coeff) for par in self.model[node]['parents']}

            self.model[node]['coeff'] = coeff

    def _linear_comb_parents(self, node, obs=None):
        linear_comb = 0.0
        if len(self.model[node]['parents']) > 0:
            for par in self.model[node]['parents']:
                if obs is None:
                    linear_comb += self.model[par]['values'] * torch.tensor(self.model[node]['coeff'][par])
                else:
                    linear_comb += torch.tensor(obs[par]) * torch.tensor(self.model[node]['coeff'][par])
        return linear_comb

    def compute_node(self, node):
        """
        sampling using structural equations
        """
        linear_comb = self._linear_comb_parents(node)
        linear_comb += self.model[node]['noise_values']
        self.model[node]['values'] = linear_comb
        return linear_comb

    def _abduct_node_par(self, node, obs, **kwargs):
        linear_comb = self._linear_comb_parents(node, obs=obs)
        noise = torch.tensor(obs[node]) - linear_comb
        return noise

    def _abduct_node_par_unobs(self, node, obs, scm_abd, **kwargs):
        """
        Abduction when one parent is not fully observed, whose parents were fully observed.
        Meaning we transform the distribution using the relationship between parents and noise
        given the invertability of the function

        p(eps=x) = p(y=g(x)) where g(x) maps x to eps given a certain observed state for the node.
        For linear model that is x_unobs = (sum beta_i parent_i - x_j + eps_j) / beta_unobs
        and eps_j = (x_j - (sum beta_i parent_i + beta_unobs x_unobs))
        """
        assert not (scm_abd is None)
        noisy_pars = [par for par in self.model[node]['parents'] if par not in obs.index]
        if len(noisy_pars) != 1:
            raise NotImplementedError('not implemented for more or less than one parent')
        else:
            raise NotImplementedError('implementation must be validated.')
            lc = 0
            for par in self.model[node]['parents']:
                if par not in noisy_pars:
                    lc += self.model[par]['coeff'] * obs[par]
            noisy_par = noisy_pars[0]
            # Note: we assume that for the scm the parents were already abducted from obs
            lc_noisy_par =  self._linear_comb_parents(noisy_par, obs=obs)
            coeff_noisy_par = self.model[node]['coeff'][noisy_par]

            def sample(scm):
                eps_j = obs[node] - (lc + coeff_noisy_par * (lc_noisy_par + scm.model[noisy_par]['noise_values']))
                return eps_j

            return sample

    def _abduct_node_obs(self, node, obs, samplesize=10**7):
        scm_sampler = self.copy()
        scm_sampler.sample_context(samplesize)
        scm_sampler.compute()
        X = scm_sampler.get_values()
        U = scm_sampler.get_noise_values()

        #mb = scm_sampler.get_markov_blanket(node)
        noise_var = 'u_' + node
        obs_vars = sorted(list(obs.index))

        gaussian_estimator = GaussianConditionalEstimator()

        train_inputs = U[noise_var].to_numpy()
        train_context = X[obs_vars].to_numpy()

        gaussian_estimator.fit(train_inputs=train_inputs,
                               train_context=train_context)

        d = gaussian_estimator.conditional_distribution_univariate(obs[obs_vars].to_numpy())
        return d