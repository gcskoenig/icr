from mcr.causality.scms import BinomialBinarySCM
import torch
import numpy as np
import torch.distributions as dist
from mcr.causality.dags import DirectedAcyclicGraph
import json
import logging
from mcr.distributions.multivariate import TransformedUniform

logger = logging.getLogger()

class SigmoidBinarySCM(BinomialBinarySCM):

    def __init__(self, dag, p_dict={}, sigmoid_nodes={}, coeff_dict={}, u_prefix='u_'):
        super(BinomialBinarySCM, self).__init__(dag, u_prefix=u_prefix)

        self.INVERTIBLE = True # not all units are invertible

        for node in self.topological_order:
            if node not in p_dict:
                p_dict[node] = torch.tensor(np.random.uniform(0, 1))
            else:
                p_dict[node] = torch.tensor(p_dict[node])

            # learn distribution object
            if node in sigmoid_nodes:
                self.model[node]['noise_distribution'] = dist.Uniform(0, 1)
                self.model[node]['sigmoidal'] = True
                assert node in coeff_dict.keys()
                self.model[node]['coeff'] = coeff_dict[node]
            else:
                self.model[node]['noise_distribution'] = dist.Binomial(probs=p_dict[node])
                self.model[node]['sigmoidal'] = False

        self.p_dict = p_dict

    def _linear_comb_parents(self, node, obs=None):
        if self.model[node]['sigmoidal']:
            # for sigmoidal do the same but shift all values by 0.5 such that centered around 0
            linear_comb = 0.0
            if len(self.model[node]['parents']) > 0:
                for par in self.model[node]['parents']:
                    assert par in self.model[node]['coeff'].keys()
                    coeff = self.model[node]['coeff'][par]
                    if obs is None:
                        linear_comb += (self.model[par]['values'] - 0.5) * coeff
                    else:
                        linear_comb += (torch.tensor(obs[par]) - 0.5) * coeff
            return linear_comb
        else:
            return super()._linear_comb_parents(node, obs=obs)

    def save(self, filepath):
        self.dag.save(filepath)
        p_dict = {}
        sigmoidal = []
        coeff_dict = {}
        for var_name in self.dag.var_names:
            if self.model[var_name]['sigmoidal']:
                sigmoidal.append(var_name)
                coeff_dict[var_name] = self.model[var_name]['coeff']
            else:
                p_dict[var_name] = self.model[var_name]['noise_distribution'].probs.item()
        scm_dict = {}
        scm_dict['p_dict'] = p_dict
        scm_dict['y_name'] = self.predict_target
        scm_dict['sigmoidal'] = sigmoidal
        scm_dict['coeff'] = coeff_dict
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
        scm = SigmoidBinarySCM(dag, p_dict=scm_dict['p_dict'], sigmoid_nodes=scm_dict['sigmoidal'],
                               coeff_dict=scm_dict['coeff'])
        scm.set_prediction_target(scm_dict['y_name'])
        # noise_vals = pd.read_csv(filepath + '_noise_vals.csv')
        return scm

    def predict_log_prob_obs(self, x_pre, y_name, y=1):
        """
        Function that predicts log probability of y given x^pre
        p(y=1|x^pre)
        """
        scm_ = self.abduct(x_pre)
        p = scm_.model[y_name]['noise_distribution'].p_y_1
        log_p = torch.log(p)
        return log_p

    def compute_node(self, node):
        """
        sampling using structural equations
        """
        linear_comb = self._linear_comb_parents(node)
        if self.model[node]['sigmoidal']:
            phi = torch.sigmoid(linear_comb)
            output = self.model[node]['noise_values'] <= phi
            return output
        else:
            return super().compute_node(node)

    def _abduct_node_par(self, node, obs, **kwargs):
        if self.model[node]['sigmoidal']:
            raise NotImplementedError('Not implemented for sigmoidal units')
        else:
            return super()._abduct_node_par(node, obs, **kwargs)

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
            if self.model[noisy_par]['sigmoidal']:
                # compute other input to the variable (except unobserved)
                linear_comb = 0
                for par in self.model[node]['parents']:
                    if par not in noisy_pars:
                        linear_comb += obs[par]

                # sampling function, assuming an scm where the noise values for the parent have been sampled
                def sample(scm):
                    if scm.model[noisy_par]['noise_values'] is None:
                        raise RuntimeError('Noise values for {} must be sampled first'.format(noisy_par))
                    noisy_par_values = scm.model[noisy_par]['noise_values']
                    noisy_par_dist = scm.model[noisy_par]['noise_distribution']
                    y = noisy_par_values <= noisy_par_dist.phi
                    values = ((linear_comb + y) % 2) == obs[node]
                    return values

                return sample
            else:
                return super()._abduct_node_par(node, obs, scm_abd, **kwargs)

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
        For unobserved nodes where all parents are observed
        """
        if self.model[node]['sigmoidal']:
            def helper(node, obs, multiply_node=False):
                # computes the conditional probability of a child given its parents for obs
                p = 1
                if multiply_node:
                    raise NotImplementedError('multiply_node not implemented in helper')
                for par in self.model[node]['children']:
                    p *= self._cond_prob_pars(par, obs).probs
                return p

            sigma = torch.sigmoid(self._linear_comb_parents(node, obs=obs))

            obs_1 = obs.copy()
            obs_1[node] = 1
            prob_1 = helper(node, obs_1) * sigma

            obs_0 = obs.copy()
            obs_0[node] = 0
            prob_0 = helper(node, obs_0) * (1 - sigma)

            p = prob_1 / (prob_0 + prob_1) # probability that y = 1 | x_pre
            # handle cases where slightly larger or smaller than bounds
            if p < 0.0:
                p = 0.0
                logger.debug("probability {} was abducted for node {} and obs {}".format(p, node, obs))
            elif p > 1.0:
                p = 1.0
                logger.debug("probability {} was abducted for node {} and obs {}".format(p, node, obs))

            return TransformedUniform(sigma, p)
        else:
            super()._abduct_node_par(node, obs, n_samples=n_samples, **kwargs)

    def predict_log_prob_individualized_obs(self, obs_pre, obs_post, intv_dict, y_name, y=1):
        """
        Individualized post-recourse prediction
        """
        assert self.model[y_name]['sigmoidal']

        phi_pre = torch.sigmoid(self._linear_comb_parents(y_name, obs=obs_pre))
        phi_post = torch.sigmoid(self._linear_comb_parents(y_name, obs=obs_post))
        phi_delta = abs(phi_post - phi_pre)

        scm_pre_abd = self.abduct(obs_pre)
        p_y_1_pre = scm_pre_abd.model[y_name]['noise_distribution'].p_y_1

        cmp1 = min(phi_post, phi_pre) * p_y_1_pre
        cmp4 = (1 - max(phi_pre, phi_post)) * (1 - p_y_1_pre)
        cmp2 = phi_delta * (1 - p_y_1_pre)
        cmp3 = phi_delta * p_y_1_pre

        y_1 = cmp1 + (phi_post > phi_pre) * cmp2
        y_0 = cmp4 + (phi_post <= phi_pre) * cmp3

        res = torch.log(y_1 / (y_1 + y_0))
        return res