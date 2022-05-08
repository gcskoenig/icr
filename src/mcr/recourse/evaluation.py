import numpy as np
import torch
import logging
import functools
from mcr.causality.utils import indvd_to_intrv

logger = logging.getLogger(__name__)


def similar(in1, in2, nr_digits=2):
    in1_r = [round(x, nr_digits) for x in in1]
    in2_r = [round(x, nr_digits) for x in in2]
    return in1_r == in2_r

class GreedyEvaluator:

    def __init__(self, scm, obs, costs, features, lbd, rounding_digits=2,
                 subpopulation_size=None, predict_log_proba=None, y_name=None,
                 multi_objective=False):
        self._subpopulation_size = subpopulation_size
        self._predict_log_proba = predict_log_proba
        self._y_name = y_name
        self.rounding_digits = rounding_digits
        self._scm, self._obs, self._costs, self._features, self._lbd = scm, obs, costs, features, lbd
        self.memoize_count = 0
        self.total_count = 0
        self.multi_objective = multi_objective

    def perc_saved(self):
        return self.memoize_count / self.total_count

    @functools.cache
    def _evaluate(self, eta, thresh, r_type, individual):
        self.memoize_count -= 1
        intv_dict = indvd_to_intrv(self._scm, self._features, individual, self._obs, causes_of=None)

        # assumes that sample_context was already called
        scm_ = self._scm.copy()

        # for subpopulation-based recourse at this point nondescendants are fixed
        if r_type == 'subpopulation':
            scm_ = scm_.fix_nondescendants(intv_dict, self._obs)
            cntxt = scm_.sample_context(self._subpopulation_size)

        # sample from intervened distribution for obs_sub
        values = scm_.compute(do=intv_dict)
        predictions = self._predict_log_proba(values[self._features])[:, 1]
        expected_above_thresh = np.mean(np.exp(predictions) >= thresh)

        ind = np.abs(np.array(individual))
        cost = np.dot(ind, self._costs)  # intervention cost
        #acceptance_cost = expected_above_thresh < eta

        return expected_above_thresh, cost

    @functools.cache
    def _evaluate_meaningful(self, gamma, r_type, individual):
        self.memoize_count -= 1
        # WARNING: for individualized recourse we expect the scm to be abducted already

        intv_dict = indvd_to_intrv(self._scm, self._features, individual, self._obs)

        # assues that sample_context was already called
        scm_ = self._scm.copy()

        # for subpopulation-based recourse at this point nondescendants are fixed
        if r_type == 'subpopulation':
            acs = scm_.dag.get_ancestors_node(self._y_name)
            intv_dict_causes = {k: intv_dict[k] for k in acs & intv_dict.keys()}
            if len(intv_dict_causes.keys()) != len(intv_dict.keys()):
                logger.debug('Intervention dict contained interventions on non-ascendants of Y ({})'.format(self._y_name))
            scm_ = scm_.fix_nondescendants(intv_dict_causes, self._obs)
            scm_.sample_context(self._subpopulation_size)

        if r_type == 'subpopulation' and len(intv_dict_causes.keys()) == 0:
            # use normal prediction to also incorporate information from effects
            perc_positive = torch.exp(self._scm.predict_log_prob_obs(self._obs, self._y_name, y=1)).item()
        else:
            # sample from intervened distribution for obs_sub
            values = scm_.compute(do=intv_dict)
            perc_positive = values[self._y_name].mean()

        # meaningfulness_cost = perc_positive < gamma
        ind = np.abs(np.array(individual))
        cost = np.dot(ind, self._costs)
        return perc_positive, cost

    def evaluate(self, eta, thresh, r_type, individual, return_split=False):
        self.memoize_count += 1
        self.total_count += 1
        individual = [round(el, self.rounding_digits) for el in individual]
        individual = tuple(individual)
        objective, cost = self._evaluate(eta, thresh, r_type, individual)
        if self.multi_objective or return_split:
            return float(objective), float(cost),
        else:
            objective = objective > eta
            return float(-cost + self._lbd * objective),

    def evaluate_meaningful(self, gamma, r_type, individual, return_split=False):
        self.memoize_count += 1
        self.total_count += 1
        individual = [round(el, self.rounding_digits) for el in individual]
        individual = tuple(individual)
        objective, cost = self._evaluate_meaningful(gamma, r_type, individual)
        if self.multi_objective or return_split:
            return float(objective), float(cost),
        else:
            objective = objective > gamma
            return float(-cost + self._lbd * objective),
