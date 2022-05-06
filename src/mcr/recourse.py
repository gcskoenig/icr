import random
import math
import numpy as np
import pandas as pd
import torch
import json

from sklearn.metrics import accuracy_score
import logging

from deap import base, creator
from deap.algorithms import eaMuPlusLambda
from deap import tools
from tqdm import tqdm

from mcr.evaluation import GreedyEvaluator, similar
from mcr.causality.utils import indvd_to_intrv

# LOGGING

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# RECOURSE FUNCTIONS



def compute_h_post_individualized(scm, X_pre, X_post, invs, features, y_name, y=1):
    """
    Computes the individualized post-recourse predictions (probabilities)
    """
    log_probs = np.zeros(invs.shape[0])
    for ix in range(invs.shape[0]):
        intv_dict = indvd_to_intrv(scm, features, invs.iloc[ix, :], X_pre.iloc[0, :])
        log_probs[ix] = torch.exp(scm.predict_log_prob_individualized_obs(X_pre.iloc[ix, :], X_post.iloc[ix, :],
                                                                          intv_dict, y_name, y=y))
    h_post_individualized = pd.DataFrame(log_probs, columns=['h_post_individualized'])
    h_post_individualized.index = X_pre.index.copy()
    return h_post_individualized

def recourse(scm_, features, obs, costs, r_type, t_type, predict_log_proba=None, y_name=None, cleanup=True, gamma=None,
             eta=None, thresh=None, lbd=1.0, subpopulation_size=500, NGEN=400, CX_PROB=0.3, MX_PROB=0.05,
             POP_SIZE=1000, rounding_digits=2, binary=False, multi_objective=False):

    evaluator = GreedyEvaluator(scm_, obs, costs, features, lbd, rounding_digits=rounding_digits,
                                subpopulation_size=subpopulation_size, predict_log_proba=predict_log_proba,
                                y_name=y_name, multi_objective=multi_objective)

    if multi_objective:
        creator.create("FitnessMin", base.Fitness, weights=(lbd, -1.0))
    else:
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    IND_SIZE = len(features)

    toolbox = base.Toolbox()
    if binary:
        toolbox.register("intervene", random.randint, 0, 1)
    else:
        toolbox.register("intervene", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.intervene, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=CX_PROB)
    if binary:
        toolbox.register("mutate", tools.mutFlipBit, indpb=MX_PROB)
    else:
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    if t_type == 'acceptance':
        assert not predict_log_proba is None
        assert not thresh is None
        toolbox.register("evaluate", evaluator.evaluate, eta, thresh, r_type)
    elif t_type == 'improvement':
        assert not y_name is None
        assert not gamma is None
        toolbox.register("evaluate", evaluator.evaluate_meaningful, gamma, r_type)
    else:
        raise NotImplementedError('only t_types acceptance or improvement are available')

    stats = tools.Statistics(key=lambda ind: np.array(ind.fitness.values))
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop = toolbox.population(n=POP_SIZE)
    if multi_objective:
        hof = tools.ParetoFront(similar)
    else:
        hof = tools.HallOfFame(max(50, round(POP_SIZE/10)))
    pop, logbook = eaMuPlusLambda(pop, toolbox, POP_SIZE, POP_SIZE * 2, CX_PROB, MX_PROB, NGEN,
                                  stats=stats, halloffame=hof, verbose=False)

    if multi_objective:
        invds = np.array(list(hof))
        perf = np.array([x.values for x in list(hof.keys)])

        min_cost_constrained = np.min(perf[perf[:, 0] > 0.95, 1])
        best_ix = np.where(perf[:, 1] == min_cost_constrained)[0][0]
        winner = invds[best_ix, :]
    else:
        winner = list(hof)[0]

    winner = [round(x, ndigits=rounding_digits) for x in winner]

    def eval_cost(winner):
        if t_type == 'acceptance':
            goal_cost, intv_cost = evaluator.evaluate(eta, thresh, r_type, winner, return_split=True)
        elif t_type == 'improvement':
            goal_cost, intv_cost = evaluator.evaluate_meaningful(gamma, r_type, winner, return_split=True)
        return goal_cost, intv_cost

    goal_cost, intv_cost = eval_cost(winner)

    # if goal could not be met return the empty intervention
    goal_met = False
    if not gamma is None:
        goal_met = goal_cost < gamma
    elif not eta is None:
        goal_met = goal_cost < eta

    if not goal_met:
        winner = [0.0 for _ in winner]
        goal_cost, intv_cost = eval_cost(winner)

    # cleanup
    if cleanup:
        del creator.FitnessMin
        del creator.Individual
        del evaluator

    return winner, pop, logbook, goal_cost, intv_cost


def recourse_population(scm, X, y, U, y_name, costs, proportion=0.5, nsamples=10 ** 4, r_type='individualized',
                        t_type='acceptance', gamma=0.7, eta=0.7, thresh=0.5, lbd=1.0, subpopulation_size=500,
                        model=None, use_scm_pred=False, predict_individualized=False, NGEN=400, POP_SIZE=1000,
                        rounding_digits=2):
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

    intv_features = X.columns
    if t_type == 'improvement':
        causes_dag = list(scm.dag.get_ancestors_node(y_name))
        causes = [nd for nd in intv_features if nd in causes_dag]  # to make sure the ordering is as desired
        ixs_causes = np.array([np.arange(len(intv_features))[cause == intv_features][0] for cause in causes])
        costs = costs[ixs_causes]
        intv_features = causes

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
        winner, pop, logbook, goal_cost, intv_cost = recourse(scm_, intv_features, obs, costs, r_type, t_type,
                                                              predict_log_proba=predict_log_proba, y_name=y_name,
                                                              gamma=gamma, eta=eta,
                                                              thresh=thresh, lbd=lbd,
                                                              subpopulation_size=subpopulation_size,
                                                              NGEN=NGEN, POP_SIZE=POP_SIZE,
                                                              rounding_digits=rounding_digits,
                                                              multi_objective=False)

        intervention = indvd_to_intrv(scm, intv_features, winner, obs)

        interventions.append(winner)
        goal_costs.append(goal_cost)
        intv_costs.append(intv_cost)

        # compute the actual outcome for this observation
        scm_true = scm.copy()
        u_tmp = U.iloc[ix, :].to_dict()
        scm_true.set_noise_values(u_tmp)
        sample = scm_true.compute(do=intervention)
        X_new.iloc[ix, :] = sample[X.columns].to_numpy()
        y_new.iloc[ix] = sample[y_name]

    logging.debug('Collecting results...')
    interventions = np.array(interventions)
    interventions = pd.DataFrame(interventions, columns=intv_features)
    interventions['ix'] = X.index[ixs_recourse]
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

    if r_type == 'individualized' and t_type == 'improvement' and predict_individualized:
        logging.debug('Computing individualized post-recourse predictions...')
        h_post_indiv = compute_h_post_individualized(scm, X_pre, X_post, interventions, intv_features, y_name, y=1)
        h_post['h_post_individualized'] = h_post_indiv['h_post_individualized']

    for df in [X_post, y_post, interventions, X_pre, y_pre, y_hat_pre, h_post]:
        df.index = X.index[ixs_recourse]

    # compute model performance on pre- and post-recourse data
    predict_pre = model.predict(X_pre)
    predict_post = model.predict(X_post)
    accuracy_pre = accuracy_score(y_pre, predict_pre)
    accuracy_post = accuracy_score(y_post, predict_post)


    logging.debug('Computing stats...')
    stats = {}
    ixs_rp = interventions[np.abs(interventions.sum(axis=1)) >= 0].index # indexes for which recourse was performed
    stats['accuracy_pre'] = accuracy_pre
    stats['accuracy_post'] = accuracy_post
    stats['recourse_seeking_ixs'] = list(interventions.index.copy())
    stats['recourse_recommended_ixs'] = list(ixs_rp.copy())
    stats['perc_recomm_found'] = float(ixs_rp.shape[0] / X_post.shape[0])
    stats['gamma'] = float(gamma)
    stats['eta'] = float(eta)
    stats['gamma_obs'] = float(y_post[ixs_rp].mean())
    stats['gamma_obs_pre'] = float(y_pre[ixs_rp].mean())
    eta_obs = (h_post.loc[ixs_rp, :] >= thresh).mean()
    stats['eta_obs'] = float(eta_obs['h_post'])
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