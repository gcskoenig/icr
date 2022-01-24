import copy
import networkx as nx
import pandas as pd
import torch
from torch import Tensor
from torch.distributions import Normal, Distribution
import numpy as np
import json

from mcr.causality import DirectedAcyclicGraph
from mcr.estimation import GaussianConditionalEstimator
from mcr.backend.dist import TransformedUniform

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

import logging

logger = logging.getLogger(__name__)

class StructuralCausalModel:
    """
    Semi-abstract class for SCM, defined by a DAG.
    Implementation based on Pyro, implements do-operator and
    tools for the computation of counterfactuals.
    """

    def __init__(self, dag: DirectedAcyclicGraph, u_prefix='u_'):
        """
        Args:
            dag: DAG, defining SCM
        """
        # Model init
        self.dag = dag
        self.topological_order = []
        self.INVERTIBLE = False
        self.u_prefix = u_prefix
        self.predict_target = None

        # the model dictionary is an internal representation of the SCM. The keys depend on the specific implementation.
        self.model = {}

        for node in nx.algorithms.dag.topological_sort(dag.DAG):
            parents = dag.DAG.predecessors(node)
            children = dag.DAG.successors(node)
            node, parents, children = dag.var_names[node], [dag.var_names[node] for node in parents], \
                                      [dag.var_names[node] for node in children]

            self.model[node] = {
                'parents': tuple(parents),  # parent variable names
                'children': tuple(children),  # children variable names
                'values': None,  # values for the variable
                'noise_values': None,  # values for the noise
                'noise_distribution': None,  # distribution for the noise, may be torch.tensor if point-mass probability
                'intervened': False # indicates whether a variables was intervened upon and must therefore be ignore during reconstruction
                # 'noise_abducted': None  # deprecated functionality
            }
            self.topological_order.append(node)

    def uname_to_name(self, u_name):
        return u_name[len(self.u_prefix):]

    def name_to_uname(self, name):
        return self.u_prefix + name

    def log_prob_u(self, u):
        raise NotImplementedError('log_prob_u not Implemented for abstract SCM class')

    def set_prediction_target(self, y_name):
        self.predict_target = y_name

    def clear_prediction_target(self):
        self.predict_target = None

    def clear_values(self):
        # remove all sampled values
        for node in self.dag.var_names:
            self.model[node]['noise_values'] = None
            self.model[node]['values'] = None

    def remove_parents(self, node):
        """removes endogenous parents for node"""
        # remove child from parents
        for parent in self.model[node]['parents']:
            children = set(self.model[parent]['children'])
            children.difference_update({node})
            self.model[parent]['children'] = tuple(children)
        # remove parents from node
        self.model[node]['parents'] = tuple([])

    def copy(self):
        """
        Creates a deepcopy of the SCM.
        """
        return copy.deepcopy(self)

    def get_sample_size(self):
        """
        Returns the current sample size as determined by the noise_values.
        """
        context_0 = self.model[self.topological_order[0]]['noise_values']
        if context_0 is None:
            return 0
        else:
            return context_0.shape[0]

    def get_values(self):
        """
        Returns the current state values in a pandas.DataFrame (endogenous variables only).
        """
        arr = torch.stack([self.model[node]['values'] for node in self.dag.var_names], dim=1).numpy()
        df = pd.DataFrame(arr, columns=self.dag.var_names)
        return df

    def get_noise_values(self):
        """
        Returns the current state values in a pandas.DataFrame (exogenous variables only).
        The noise variables are named u_[var_name].
        """
        arr = torch.stack([self.model[node]['noise_values'] for node in self.dag.var_names], dim=1).numpy()
        df = pd.DataFrame(arr, columns=[self.name_to_uname(var_name) for var_name in self.dag.var_names])
        return df

    def set_noise_values(self, dict, return_values=True):
        """
        Set the noise values from a dictionary like objects (such as pandas.DataFrame).
        Naming convention: noise variables are named u_[var_name]
        """
        for unode in dict.keys():
            node = self.uname_to_name(unode)  # remove u_ from the name
            value = dict[unode]
            if not type(value) is torch.Tensor:
                value = torch.tensor(value)
            self.model[node]['noise_values'] = torch.reshape(value, (-1,))
        if return_values:
            return self.get_noise_values()

    def get_noise_distributions(self, check_deterministic=False, return_intervened=False):
        """
        If check deterministic we check whether all noise distributions are torch.tensor values (static)
        """
        noises = {}
        deterministic = True
        for node in self.dag.var_names:
            if not self.model[node]['intervened'] or return_intervened:
                noise = self.model[node]['noise_distribution']
                u_node = self.name_to_uname(node)
                noises[u_node] = noise
                if check_deterministic:
                    deterministic = deterministic and (type(noise) is torch.Tensor)

        # raise error if not deterministic if check_eterministic
        if check_deterministic and not deterministic:
            raise RuntimeError('Noise terms are not deterministic')

        return noises

    def get_markov_blanket(self, node: str) -> set:
        """
        Get the markov blanket variables for a node as set of variable names.
        """
        return self.dag.get_markov_blanket(node)

    def sample_context(self, size: int, seed=None):
        """
        Use the noise in self.model to generate noise.values in the model.
        Either a torch.Distribution object, a torch.tensor for point-mass distributions or a callable function.
        """
        for node in self.topological_order:
            if isinstance(self.model[node]['noise_distribution'], Distribution):
                self.model[node]['noise_values'] = self.model[node]['noise_distribution'].sample((size,)).flatten()
            elif isinstance(self.model[node]['noise_distribution'], Tensor):
                self.model[node]['noise_values'] = self.model[node]['noise_distribution'].repeat(size)
            elif callable(self.model[node]['noise_distribution']):  # TODO: document this case better.
                self.model[node]['noise_values'] = self.model[node]['noise_distribution'](self)
            else:
                raise NotImplementedError('The noise is neither a torch.distributions.Distribution nor torch.Tensor')
        return self.get_noise_values()

    def do(self, intervention_dict, verbose=False):
        """Intervention

        :param intervention_dict: dictionary of interventions of the form 'variable-name' : value
        :return: copy of the structural causal model with the performend interventions
        """
        scm_itv = self.copy()
        if verbose:
            logging.info('Intervening on nodes: {}'.format(intervention_dict.keys()))
        # update structural equations
        for node in intervention_dict.keys():
            scm_itv.remove_parents(node)
            scm_itv.model[node]['noise_distribution'] = torch.tensor(intervention_dict[node])
            scm_itv.model[node]['intervened'] = True
        scm_itv.clear_values()
        scm_itv.dag.do(intervention_dict.keys())
        return scm_itv

    def fix_nondescendants(self, intervention_dict, obs):
        """Fix the values of nondescendants of the intervened upon-variables that have been observed

        :param intervention_dict: dictionary of interventions of the form 'variable-name' : value
        :param obs: dictionary-like observation (individual)
        :return: copy of the scm where nondescendants are fixed using do
        """
        nds = self.dag.get_nondescendants(intervention_dict.keys())
        nds = set.intersection(set(obs.keys()), nds)
        scm_ = self.do(obs[nds])
        return scm_

    def abduct_node(self, node, obs, scm_partially_abducted=None, **kwargs):
        """Abduction

        Args:
            node: name of the node
            obs: observation tensor?
            scm_partially_abducted: all parents of the node must have been abducted already.

        Returns:
            object to be stored as "noise" for the node in the model dictionary.
        """

        logger.debug('Abducting noise for: {}'.format(node))
        # check whether node was observed, then only parents necessary for deterministic reconstruction
        if node in obs.index and self.INVERTIBLE:
            # if parents observed then reconstruct deterministically
            if set(self.model[node]['parents']).issubset(set(obs.index)):
                logger.debug('\t...by inverting the structural equation given the parents and node.')
                return self._abduct_node_par(node, obs, **kwargs)
            # if not all parents observed then the noise is dependent on the unobserved parent
            else:
                logger.debug('\t...as function of the parents noise.')
                return self._abduct_node_par_unobs(node, obs, scm_partially_abducted, **kwargs)
        elif node not in obs.index:
            logger.debug('\t...using analytical formula and MC integration.')
            return self._abduct_node_obs(node, obs, **kwargs)
        else:
            raise NotImplementedError('No solution for variable observed but not invertible developed yet.')

    def abduct(self, obs, **kwargs):
        """ Abduct all variables from observation
        Assumes a topological ordering in the DAG.
        returns a separate SCM where the abduction was performed.
        """
        scm_abd = self.copy()
        for node in self.topological_order:
            scm_abd.model[node]['noise_distribution'] = self.abduct_node(node, obs, scm_partially_abducted=scm_abd,
                                                                         **kwargs)
            scm_abd.model[node]['noise_values'] = None
            scm_abd.model[node]['values'] = None
        return scm_abd

    def compute_node(self, node):
        """
        sampling using structural equations
        """
        raise NotImplementedError('not implemented in semi-abstract class')

    def compute(self, do={}):
        """
        Returns a sample from SEM (observational distribution), columns are ordered as var self.dag.var_names in DAG
        Requires that context variables are set/sampled.
        Args:
            do: dictionary with {'node': value}

        Returns: torch.Tensor with sampled value of shape (size, n_vars))
        """
        assert set(do.keys()).issubset(set(self.dag.var_names))
        assert self.get_sample_size() > 0
        for node in self.topological_order:
            if node in do.keys():
                self.model[node]['values'] = torch.tensor(do[node]).repeat(self.get_sample_size())
            else:
                self.model[node]['values'] = self.compute_node(node).reshape(-1)
            assert (~self.model[node]['values'].isnan()).all()
        return self.get_values()


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

    def _get_pyro_model(self, target_node):
        """
        Returns pyro model where the target node is modeled as deterministic function of
        a probabilistic noise term, whereas all other nodes are directly modeled as
        probabilistic variables (such that other variables can be observed and the
        noise term can be inferred).
        """
        def pyro_model():
            var_dict = {}
            for node in self.topological_order:
                input = torch.tensor(0.0)
                for par in self.model[node]['parents']:
                    input = input + var_dict[par]
                input = torch.remainder(input, 2)
                if node != target_node:
                    prob = (1.0 - input) * self.p_dict[node] + input * (1 - self.p_dict[node])
                    var = pyro.sample(node, dist.Binomial(probs=prob))
                    var_dict[node] = var.flatten()
                else:
                    noise = pyro.sample('u_'+node, dist.Binomial(probs=self.p_dict[node]))
                    var = torch.remainder(input + noise, 2)
                    var_dict[node] = var.flatten()
            return var_dict

        return pyro_model

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