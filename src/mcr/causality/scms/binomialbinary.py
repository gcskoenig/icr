import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
import numpy as np
import json
from mcr.causality import DirectedAcyclicGraph
from mcr.causality.scms import StructuralCausalModel
import torch.distributions as dist
import logging

logger = logging.getLogger(__name__)


class BinomialBinarySCM(StructuralCausalModel):

    def __init__(self, dag, p_dict={}, u_prefix='u_'):
        super(BinomialBinarySCM, self).__init__(dag, u_prefix=u_prefix)

        self.INVERTIBLE = True

        for node in self.topological_order:
            if node not in p_dict:
                p_dict[node] = torch.tensor(np.random.uniform(0, 1))
            else:
                p_dict[node] = torch.tensor(p_dict[node])
            self.model[node]['noise_distribution'] = dist.Binomial(probs=p_dict[node])

        self.p_dict = p_dict

    def save(self, filepath):
        self.dag.save(filepath)
        p_dict = {}
        for var_name in self.dag.var_names:
            p_dict[var_name] = self.model[var_name]['noise_distribution'].probs.item()
        scm_dict = {}
        scm_dict['p_dict'] = p_dict
        scm_dict['y_name'] = self.predict_target
        try:
            with open(filepath + '_p_dict.json', 'w') as f:
                json.dump(scm_dict, f)
        except Exception as exc:
            logging.warning('Could not save p_dict.json')
            logging.info('Exception: {}'.format(exc))

    @staticmethod
    def load(filepath):
        dag = DirectedAcyclicGraph.load(filepath)
        f = open(filepath + '_p_dict.json')
        scm_dict = json.load(f)
        f.close()
        scm = BinomialBinarySCM(dag, scm_dict['p_dict'])
        scm.set_prediction_target(scm_dict['y_name'])
        # noise_vals = pd.read_csv(filepath + '_noise_vals.csv')
        return scm

    # def _get_pyro_model(self, target_node):
    #     """
    #     Returns pyro model where the target node is modeled as deterministic function of
    #     a probabilistic noise term, whereas all other nodes are directly modeled as
    #     probabilistic variables (such that other variables can be observed and the
    #     noise term can be inferred).
    #     """
    #     def pyro_model():
    #         var_dict = {}
    #         for node in self.topological_order:
    #             input = torch.tensor(0.0)
    #             for par in self.model[node]['parents']:
    #                 input = input + var_dict[par]
    #             input = torch.remainder(input, 2)
    #             if node != target_node:
    #                 prob = (1.0 - input) * self.p_dict[node] + input * (1 - self.p_dict[node])
    #                 var = pyro.sample(node, dist.Binomial(probs=prob))
    #                 var_dict[node] = var.flatten()
    #             else:
    #                 noise = pyro.sample('u_'+node, dist.Binomial(probs=self.p_dict[node]))
    #                 var = torch.remainder(input + noise, 2)
    #                 var_dict[node] = var.flatten()
    #         return var_dict
    #
    #     return pyro_model

    def _linear_comb_parents(self, node, obs=None):
        linear_comb = 0.0
        if len(self.model[node]['parents']) > 0:
            for par in self.model[node]['parents']:
                if obs is None:
                    linear_comb += self.model[par]['values']
                else:
                    linear_comb += torch.tensor(obs[par])
        return linear_comb

    def log_prob_u(self, u):
        """
        For a dictionary-like object u we compute the joint probability
        """
        scm_ = self.copy()
        scm_.clear_values()
        scm_.set_noise_values(u, return_values=False)

        log_p = 0
        for u_nd in u.keys():
            nd = self.uname_to_name(u_nd)
            value = u[u_nd]
            p_nd = None
            if not type(value) is Tensor:
                value = torch.tensor(value)
            dist = scm_.model[nd]['noise_distribution']
            if isinstance(dist, Distribution):
                p_nd = dist.log_prob(value)
            elif isinstance(dist, Tensor):
                p_nd = torch.log(value == dist)
            elif callable(dist):
                # attention: we can only do this because the relationship is deterministic. cannot be
                # easility transferred to situations with non-invertible structural equations.
                dist_sample = dist(scm_)
                p_nd = torch.log(value == dist_sample[0])
            else:
                raise RuntimeError('distribution type not understood: dist is {}'.format(dist))

            log_p += p_nd
        return log_p

    def predict_log_prob_obs(self, x_pre, y_name, y=1):
        """
        Function that predicts log probability of y given x^pre
        p(y=1|x^pre)
        """
        scm_ = self.abduct(x_pre)
        obs_full = x_pre.copy()
        obs_full[y_name] = y
        scm__ = self.abduct(obs_full)
        u = scm__.get_noise_distributions(check_deterministic=True)
        u_y_name = self.name_to_uname(y_name)
        u_y = {u_y_name: u[u_y_name]}
        log_p = scm_.log_prob_u(u_y)
        return log_p

    def predict_log_prob(self, X_pre, y_name=None):
        """
        expects vector input
        returns 2d numpy array with order [0, 1]
        """
        if y_name is None:
            assert not self.predict_target is None
            y_name = self.predict_target
        result = np.zeros((X_pre.shape[0], 2))
        for ii in range(X_pre.shape[0]):
            log_p_1 = self.predict_log_prob_obs(X_pre.iloc[ii, :], y_name, y=1)
            result[ii, 1] = log_p_1.item()
            result[ii, 0] = torch.log(1 - torch.exp(log_p_1)).item()
        return result

    def predict_log_prob_individualized_obs(self, obs_pre, obs_post, intv_dict, y_name, y=1):
        """
        Individualized post-recourse prediction
        """
        scm_int = self.do(intv_dict)
        scm_pre_abd = self.abduct(obs_pre)
        obs_post_ = obs_post.copy()
        ys = [0, 1]

        log_probs = {}

        for y_tmp in ys:
            obs_post_[y_name] = y_tmp
            scm_int_abd = scm_int.abduct(obs_post_)

            # extract abducted values u' as dictionary
            u = scm_int_abd.get_noise_distributions(check_deterministic=True)

            ## compute their joint probability as specified by scm_
            log_probs[y_tmp] = scm_pre_abd.log_prob_u(u)

        denom = torch.log(sum([torch.exp(log_probs[y_tmp]) for y_tmp in ys]))
        res = log_probs[y] - denom
        return res

    def compute_node(self, node):
        """
        sampling using structural equations
        """
        linear_comb = self._linear_comb_parents(node)
        linear_comb += self.model[node]['noise_values']
        self.model[node]['values'] = linear_comb
        return torch.remainder(linear_comb, 2)

    def _abduct_node_par(self, node, obs, **kwargs):
        linear_comb = self._linear_comb_parents(node, obs=obs)
        noise = torch.tensor(obs[node]) - linear_comb
        return torch.remainder(noise, 2)

    def _abduct_node_par_unobs(self, node, obs, scm_abd, **kwargs):
        """
        Abduction when one parent is not fully observed, whose parents were fully observed.
        Meaning we transform the distribution using the relationship between parents and noise
        given the invertability of the function

        p(eps=x) = p(y=g(x)) where g(x) maps x to eps given a certain observed state for the node.
        For the binary model
        eps_j = (x_j - (sum parent_i + x_unobs)) % 2
         = (x_j - (sum_parent_i + sum_parent_unobs + epsilon_unobs)) % 2
        -> noise flipped if (x_j - (sum parent_i + sum_parent_unobs)) is 1
        """
        assert not (scm_abd is None)
        noisy_pars = [par for par in self.model[node]['parents'] if par not in obs.index]
        if len(noisy_pars) != 1:
            raise NotImplementedError('not implemented for more or less than one parent')
        else:
            noisy_par = noisy_pars[0]
            # Note: we assume that for the scm the parents were already abducted from obs
            # compute whether the eps_noisy_par must be flipped or is identical to eps_node
            linear_comb = 0
            for par in self.model[node]['parents']:
                if par not in noisy_pars:
                    linear_comb += obs[par]
            linear_comb_noisy_par = self._linear_comb_parents(noisy_par, obs=obs)
            flip = torch.remainder(obs[node] - linear_comb - linear_comb_noisy_par, 2)
            # transform noise to the variable distribution (assuming the respective parents were observed)
            def sample(scm):
                if scm.model[noisy_par]['noise_values'] is None:
                    raise RuntimeError('Noise values for {} must be sampled first'.format(noisy_par))
                noisy_par_values = scm.model[noisy_par]['noise_values']
                values = flip * (1-noisy_par_values) + (1-flip) * noisy_par_values
                return values
            return sample

    def _cond_prob_pars(self, node, obs):
        obs_dict = obs.to_dict()
        assert set(self.model[node]['parents']).issubset(obs_dict.keys())
        assert node in obs_dict.keys()
        input = 0
        for par in self.model[node]['parents']:
            input += obs[par]
        u_node = torch.tensor((obs[node] - input) % 2, dtype=torch.float)
        p = self.model[node]['noise_distribution'].probs
        p_new = u_node * p + (1 - u_node) * (1 - p)
        if type(p_new) is not torch.Tensor:
            p_new = torch.tensor(p_new)
        return dist.Binomial(probs=p_new)

    def _abduct_node_obs(self, node, obs, n_samples=10**3, **kwargs):
        """
        Using formula to analytically compute the distribution
        """
        def helper(node, obs, multiply_node=True):
            p = 1
            if multiply_node:
                p *= self._cond_prob_pars(node, obs).probs
            for par in self.model[node]['children']:
                p *= self._cond_prob_pars(par, obs).probs
            return p

        obs_nom = obs.copy()
        obs_nom[node] = 1
        nominator = helper(node, obs_nom)

        # get conditional distribution of node=1 | parents
        #cond_dist = self._cond_prob_pars(node, obs_nom)
        #sample = cond_dist.sample((n_samples,)).flatten()
        denominator = 0
        for ii in range(2):
            obs_ii = obs.copy()
            obs_ii[node] = ii
            denominator += helper(node, obs_ii, multiply_node=True)

        p = nominator/denominator

        # determine whether the p needs to be flipped or not to be appropriate for eps and not just y
        linear_comb = torch.remainder(self._linear_comb_parents(node, obs), 2)
        p = (1 - linear_comb) * p + linear_comb * (1 - p)

        # handle cases where slightly larger or smaller than bounds
        if p < 0.0:
            p = 0.0
            logger.debug("probability {} was abducted for node {} and obs {}".format(p, node, obs))
        elif p > 1.0:
            p = 1.0
            logger.debug("probability {} was abducted for node {} and obs {}".format(p, node, obs))

        return dist.Binomial(probs=p)