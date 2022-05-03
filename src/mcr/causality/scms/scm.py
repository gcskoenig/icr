import copy
import random

import networkx as nx
import pandas as pd
import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
import numpy as np
import json
import jax.random as jrandom
import jax.numpy as jnp

from mcr.causality import DirectedAcyclicGraph
from mcr.estimation import GaussianConditionalEstimator
from mcr.backend.dist import TransformedUniform, MultivariateIndependent

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, MixedHMC

import logging

logger = logging.getLogger(__name__)

class StructuralCausalModel:
    """
    Semi-abstract class for SCM, defined by a DAG.
    Implementation based on Pyro, implements do-operator and
    tools for the computation of counterfactuals.
    """

    def __init__(self, dag: DirectedAcyclicGraph, costs=None, y_name=None, u_prefix='u_'):
        """
        Args:
            dag: DAG, defining SCM
        """
        # Model init
        self.dag = dag
        self.topological_order = []
        self.INVERTIBLE = False
        self.u_prefix = u_prefix
        self.predict_target = y_name
        if costs is None:
            costs = list(np.ones(len(dag.var_names)) - 1)
        self.costs = costs

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

    def get_values(self, var_names=None):
        """
        Returns the current state values in a pandas.DataFrame (endogenous variables only).
        """
        if var_names is None:
            var_names = self.dag.var_names
        if len(var_names) > 0:
            arr = torch.stack([self.model[node]['values'] for node in var_names], dim=1).numpy()
            df = pd.DataFrame(arr, columns=var_names)
            return df
        else:
            return pd.DataFrame([])

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
            d = self.model[node]['noise_distribution']
            # if isinstance(d, numpyro.distributions.Delta):
            #     pass
            if d is None:
                pass
            elif isinstance(d, numpyro.distributions.Distribution) and len(d.event_shape) > 0:
                # TODO check whether right assignment
                rng_key = jrandom.PRNGKey(random.randint(0, 2 ** 8))
                vals = np.array(d.sample(rng_key, (size,)))
                self.model[node]['noise_values'] = torch.tensor(vals[:, 0])
                for jj in range(len(self.model[node]['children'])):
                    ch = self.model[node]['children'][jj]
                    self.model[ch]['noise_values'] = torch.tensor(vals[:, jj+1])
            elif isinstance(d, Distribution):
                self.model[node]['noise_values'] = self.model[node]['noise_distribution'].sample((size,)).flatten()
            elif isinstance(d, numpyro.distributions.Distribution):
                rng_key = jrandom.PRNGKey(random.randint(0, 2**8))
                vals = self.model[node]['noise_distribution'].sample(rng_key, (size,)).flatten()
                self.model[node]['noise_values'] = torch.tensor(np.array(vals))
            elif isinstance(d, Tensor):
                self.model[node]['noise_values'] = self.model[node]['noise_distribution'].repeat(size)
            elif callable(d):  # TODO: document this case better.
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

    def abduct_node(self, node, obs, scm_partially_abducted=None, infer_type='mcmc', **kwargs):
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
        pars_observed = set(self.model[node]['parents']).issubset(set(obs.index))

        # NODE OBSERVED AND INVERTIBLE
        if node in obs.index and self.INVERTIBLE:
            # if parents observed then reconstruct deterministically
            if pars_observed:
                logger.debug('\t...by inverting the structural equation | x_pa, x_j.')
                return self._abduct_node_par(node, obs, **kwargs)
            # if not all parents observed then the noise is dependent on the unobserved parent
            else:
                logger.debug('\t...as function of the parents noise.')
                return self._abduct_node_par_unobs(node, obs, scm_partially_abducted, **kwargs)

        # NODE OBSERVED BUT NOT INVERTIBLE
        elif node in obs.index and not self.INVERTIBLE:
            if pars_observed:
                if infer_type == 'svi':
                    logger.debug('\t...using svi | x_pa, x_j')
                    return self._abduct_node_par_svi(node, obs, **kwargs)
                elif infer_type == 'mcmc':
                    logger.debug('\t...using mcmc | x_pa, x_j')
                    return self._abduct_node_par_mcmc(node, obs, **kwargs)
                else:
                    raise NotImplementedError('type not implemented')
            else:
                logger.debug('\t...as function of the parents noise')
                return self._abduct_node_par_unobs(node, obs, **kwargs)

        # NODE NOT OBSERVED AND INVERTIBLE
        elif node not in obs.index and self.INVERTIBLE:
            logger.debug('\t...using analytical formula and MC integration.')
            return self._abduct_node_obs(node, obs, **kwargs)

        # NODE NOT OBSERVED AND NOT INVERTIBLE
        elif node not in obs.index and not self.INVERTIBLE:
            if infer_type == 'svi':
                logger.debug('\t...using svi | x_pa, x_ch, x_pa(ch)')
                return self._abduct_node_obs_svi(node, obs, **kwargs)
            elif infer_type == 'mcmc':
                logger.debug('\t...using mcmc | x_pa, x_ch, x_pa(ch)')
                return self._abduct_node_obs_mcmc(node, obs, **kwargs)
            else:
                raise NotImplementedError('type not implemented')
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

    def predict_log_prob_obs(self, x_pre, y_name, y=1):
        """P(Y=y|X=x_pre)"""
        raise NotImplementedError('Not implemented in abstract class')

    def predict_log_prob_individualized_obs(self, obs_pre, obs_post, intv_dict, y_name, y=1):
        """P(Y_post = y | x_pre, x_post)"""
        raise NotImplementedError('Not implemented in abstract class.')