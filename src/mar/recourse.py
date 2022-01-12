import random
import math
import numpy as np
import pandas as pd
import torch
import json

import logging

from deap import base, creator
from deap.algorithms import eaMuPlusLambda
from deap import tools
from tqdm import tqdm

# LOGGING

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# RECOURSE FUNCTIONS

def indvd_to_intrv(scm, features, individual, obs, causes_of=None):
    """
    If causes_of is None, then all interventions are added to the dictinoary.
    If it is a specific node, then only causes of that node are added.
    """
    dict = {}

    # build causes set
    causes = None
    if causes_of is None:
        causes = set(features)
    else:
        causes = scm.dag.get_ancestors_node(causes_of)

    # iterate over variables to add causes
    for ii in range(len(features)):
        var_name = features[ii]
        if individual[ii] and (var_name in causes):
            dict[var_name] = (obs[var_name] + individual[ii]) % 2
    return dict


def compute_h_post_individualized(scm, X_pre, X_post, invs, y_name, y=1):
    """
    Computes the individualized post-recourse predictions (proababilities)
    """
    log_probs = np.zeros(invs.shape[0])
    for ix in range(invs.shape[0]):
        intv_dict = indvd_to_intrv(scm, X_pre.columns, invs.iloc[ix, :], X_pre.iloc[0, :])
        log_probs[ix] = torch.exp(scm.predict_log_prob_individualized_obs(X_pre.iloc[ix, :], X_post.iloc[ix, :],
                                                                          intv_dict, y_name, y=y))
    h_post_individualized = pd.DataFrame(log_probs, columns=['h_post_individualized'])
    h_post_individualized.index = X_pre.index.copy()
    return h_post_individualized


def evaluate(predict_log_proba, thresh, eta, scm, obs, features, costs, lbd, r_type, subpopulation_size, individual,
             return_split_cost=False):
    intv_dict = indvd_to_intrv(scm, features, individual, obs)

    scm_ = scm.copy()

    # for subpopulation-based recourse at this point nondescendants are fixed
    if r_type == 'subpopulation':
        scm_ = scm_.fix_nondescendants(intv_dict, obs)
        cntxt = scm_.sample_context(subpopulation_size)

    # sample from intervened distribution for obs_sub
    values = scm_.compute(do=intv_dict)
    predictions = predict_log_proba(values[features])[:, 1]
    expected_above_thresh = np.mean(np.exp(predictions) >= thresh)

    ind = np.array(individual)
    cost = np.dot(ind, costs) # intervention cost
    acceptance_cost = expected_above_thresh < eta
    res = cost + lbd * acceptance_cost

    if return_split_cost:
        return acceptance_cost, cost
    else:
        return res,


def evaluate_meaningful(y_name, gamma, scm, obs, features, costs, lbd, r_type, subpopulation_size, individual,
                        return_split_cost=False):
    # WARNING: for individualized recourse we expect the scm to be abducted already

    intv_dict = indvd_to_intrv(scm, features, individual, obs)
    scm_ = scm.copy()

    # for subpopulation-based recourse at this point nondescendants are fixed
    if r_type == 'subpopulation':
        acs = scm_.dag.get_ancestors_node(y_name)
        intv_dict_causes = {k : intv_dict[k] for k in acs & intv_dict.keys()}
        if len(intv_dict_causes.keys()) != len(intv_dict.keys()):
            logger.debug('Intervention dict contained interventions on non-ascendants of Y ({})'.format(y_name))
        scm_ = scm_.fix_nondescendants(intv_dict_causes, obs)
        scm_.sample_context(subpopulation_size)

    perc_positive = None
    if r_type == 'subpopulation' and len(intv_dict_causes.keys()) == 0:
        # use normal prediction to also incorporate information from effects
        perc_positive = torch.exp(scm.predict_log_prob_obs(obs, y_name, y=1)).item()
    else:
        # sample from intervened distribution for obs_sub
        values = scm_.compute(do=intv_dict)
        perc_positive = values[y_name].mean()

    meaningfulness_cost = perc_positive < gamma

    ind = np.array(individual)
    cost = np.dot(ind, costs)
    res = cost + lbd * meaningfulness_cost
    if return_split_cost:
        return meaningfulness_cost, cost
    else:
        return res,


def recourse(scm_, features, obs, costs, r_type, t_type, predict_log_proba=None, y_name=None, cleanup=True, gamma=None,
             eta=None, thresh=None, lbd=1.0, subpopulation_size=100):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    IND_SIZE = len(features)
    CX_PROB = 0.3
    MX_PROB = 0.05
    NGEN = 100

    toolbox = base.Toolbox()
    toolbox.register("intervene", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.intervene, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=CX_PROB)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MX_PROB)
    toolbox.register("select", tools.selNSGA2)

    if t_type == 'acceptance':
        assert not predict_log_proba is None
        assert not thresh is None
        toolbox.register("evaluate", evaluate, predict_log_proba, thresh, eta, scm_, obs, features, costs, lbd, r_type,
                         subpopulation_size)
    elif t_type == 'improvement':
        assert not y_name is None
        assert not gamma is None
        toolbox.register("evaluate", evaluate_meaningful, y_name, gamma, scm_, obs, features, costs, lbd, r_type,
                         subpopulation_size)
    else:
        raise NotImplementedError('only t_types acceptance or improvement are available')

    stats = tools.Statistics(key=lambda ind: np.array(ind.fitness.values))
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(4)
    pop, logbook = eaMuPlusLambda(pop, toolbox, 10, 30, CX_PROB, MX_PROB, NGEN, stats=stats, halloffame=hof,
                                  verbose=False)

    winner = list(hof)[0]

    goal_cost, intv_cost = None, None
    if t_type == 'acceptance':
        goal_cost, intv_cost = evaluate(predict_log_proba, thresh, eta, scm_, obs, features, costs, lbd, r_type,
                                        subpopulation_size, winner, return_split_cost=True)
    elif t_type == 'improvement':
        goal_cost, intv_cost = evaluate_meaningful(y_name, gamma, scm_, obs, features, costs, lbd, r_type,
                                                   subpopulation_size, winner, return_split_cost=True)

    if cleanup:
        del creator.FitnessMin
        del creator.Individual

    return winner, pop, logbook, goal_cost, intv_cost


def recourse_population(scm, X, y, U, y_name, costs, proportion=0.5, nsamples=10 ** 2, r_type='individualized',
                        t_type='acceptance', gamma=0.7, eta=0.7, thresh=0.5, lbd=1.0, subpopulation_size=100,
                        model=None, use_scm_pred=False):
    assert not (model is None and not use_scm_pred)

    # initializing prediction setup
    predict_log_proba = None
    if use_scm_pred:
        scm.set_prediction_target(y_name)
        predict_log_proba = scm.predict_log_prob
    else:
        predict_log_proba = model.predict_log_proba

    logging.debug('Determining rejected individuals and individuals determined to implement recourse...')
    predictions = np.exp(predict_log_proba(X)[:, 1]).flatten() >= thresh
    ixs_rejected = np.arange(len(predictions))[predictions == 0]
    ixs_recourse = np.random.choice(ixs_rejected, size=math.floor(proportion * len(ixs_rejected)), replace=False)
    logging.debug('Detected {} rejected and {} recourse seeking individuals...'.format(len(ixs_rejected),
                                                                                       len(ixs_recourse)))

    X_new = X.copy()
    y_new = None
    if y is not None:
        y_new = y.copy()
    interventions = []
    goal_costs = []
    intv_costs = []

    logging.debug('Iterating through {} individuals to suggest recourse...'.format(len(ixs_recourse)))
    for ix in tqdm(ixs_recourse):
        obs = X.iloc[ix, :]

        scm_ = None

        # for individualized recourse abduction is performed at this step
        if r_type == 'subpopulation':
            scm_ = scm.copy()
        elif r_type == 'individualized':
            scm_ = scm.abduct(obs, n_samples=nsamples)
        else:
            raise NotImplementedError('r_type must be in {}'.format(['individualized', 'subpopulation']))

        # compute optimal action
        cntxt = scm_.sample_context(size=nsamples)
        winner, pop, logbook, goal_cost, intv_cost = recourse(scm_, X.columns, obs, costs, r_type, t_type,
                                                              predict_log_proba=predict_log_proba, y_name=y_name,
                                                              gamma=gamma, eta=eta,
                                                              thresh=thresh, lbd=lbd,
                                                              subpopulation_size=subpopulation_size)

        intervention = indvd_to_intrv(scm, X.columns, winner, obs)

        interventions.append(winner)
        goal_costs.append(goal_cost)
        intv_costs.append(intv_cost)

        # compute the actual outcome for this observation
        scm_true = scm.copy()
        scm_true.set_noise_values(U.iloc[ix, :].to_dict())
        #scm_true.sample_context(size=1)
        sample = scm_true.compute(do=intervention)
        X_new.iloc[ix, :] = sample[X.columns].to_numpy()
        y_new.iloc[ix] = sample[y_name]

    logging.debug('Collecting results...')
    interventions = np.array(interventions)
    interventions = pd.DataFrame(interventions, columns=X.columns)
    interventions['ix'] = ixs_recourse
    interventions.set_index('ix', inplace=True)

    costss = np.array([goal_costs, intv_costs]).T
    costss = pd.DataFrame(costss, columns=['goal_cost', 'intv_cost'])
    costss.index = interventions.index.copy()

    X_pre = X.iloc[ixs_recourse, :]
    y_pre = y.iloc[ixs_recourse]
    X_post = X_new.iloc[ixs_recourse, :]
    y_post = y_new.iloc[ixs_recourse]

    logging.debug('Collecting pre- and post-recourse model predictions...')
    y_hat_pre = pd.DataFrame(predictions[ixs_recourse], columns=['y_hat'])
    h_post = predict_log_proba(X_post[X.columns])[:, 1].flatten()
    h_post = pd.DataFrame(h_post, columns=['h_post'])
    h_post['h_post_individualized'] = np.nan
    h_post = np.exp(h_post)
    h_post.index = y_post.index.copy()

    if r_type == 'individualized' and t_type == 'improvement':
        logging.debug('Computing individualized post-recourse predictions...')
        h_post_indiv = compute_h_post_individualized(scm, X_pre, X_post, interventions, y_name, y=1)
        h_post['h_post_individualized'] = h_post_indiv['h_post_individualized']

    for df in [X_post, y_post, interventions, X_pre, y_pre, y_hat_pre, h_post]:
        df.index = ixs_recourse

    logging.debug('Computing stats...')
    stats = {}
    ixs_rp = interventions[interventions.sum(axis=1) >= 1].index # indexes for which recourse was performed
    stats['recourse_seeking_ixs'] = list(interventions.index.copy())
    stats['recourse_recommended_ixs'] = list(ixs_rp.copy())
    stats['perc_recomm_found'] = float(ixs_rp.shape[0] / X_post.shape[0])
    stats['gamma'] = float(gamma)
    stats['eta'] = float(eta)
    stats['gamma_obs'] = float(y_post[ixs_rp].mean())
    stats['gamma_obs_pre'] = float(y_pre[ixs_rp].mean())
    eta_obs = (h_post.loc[ixs_rp, :] >= thresh).mean()
    stats['eta_obs'] = float(eta_obs['h_post'])
    stats['eta_obs_individualized'] = float(eta_obs['h_post_individualized'])
    stats['costs'] = list(costs)  # costs for the interventions (list with len(X.columns) indexes)
    stats['lbd'] = float(lbd)
    stats['thresh'] = float(thresh)
    stats['r_type'] = str(r_type)
    stats['t_type'] = str(t_type)

    if not h_post['h_post_individualized'].hasnans:
        stats['eta_obs_individualized'] = eta_obs['h_post_individualized']
    else:
        stats['eta_obs_individualized'] = np.nan

    logging.debug('Done.')
    return U, X_pre, y_pre, y_hat_pre, interventions, X_post, y_post, h_post, costss, stats


def save_recourse_result(savepath_exp, result_tupl):
    U, X_pre, y_pre, y_hat_pre, invs, X_post, y_post, h_post, costss, stats = result_tupl
    U.to_csv(savepath_exp + 'U.csv')
    X_pre.to_csv(savepath_exp + 'X_pre.csv')
    y_pre.to_csv(savepath_exp + 'y_pre.csv')
    y_hat_pre.to_csv(savepath_exp + 'y_hat_pre.csv')
    invs.to_csv(savepath_exp + 'invs.csv')
    X_post.to_csv(savepath_exp + 'X_post.csv')
    y_post.to_csv(savepath_exp + 'y_post.csv')
    h_post.to_csv(savepath_exp + 'h_post.csv')
    costss.to_csv(savepath_exp + 'costss.csv')

    try:
        with open(savepath_exp + 'stats.json', 'w') as f:
            json.dump(stats, f)
    except Exception as exc:
        logging.warning('stats.json could not be saved.')
        logging.info('Exception: {}'.format(exc))