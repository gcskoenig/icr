import random
import math
import numpy as np

import logging

from deap import base, creator
from deap.algorithms import eaMuPlusLambda
from deap import tools
from tqdm import tqdm

# LOGGING

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# RECOURSE FUNCTIONS

def indvd_to_intrv(features, individual, obs):
    dict = {}
    for ii in range(len(features)):
        if individual[ii]:
            dict[features[ii]] = (obs[features[ii]] + individual[ii]) % 2
    return dict


def evaluate(model, thresh, scm, obs, features, costs, lbd, r_type, individual):
    intv_dict = indvd_to_intrv(features, individual, obs)

    scm_ = scm.copy()

    # for subpopulation-based recourse at this point nondescendants are fixed
    if r_type == 'subpopulation':
        scm_ = scm_.fix_nondescendants(intv_dict, obs)
        cntxt = scm_.sample_context(scm.get_sample_size())

    # sample from intervened distribution for obs_sub
    values = scm_.compute(do=intv_dict)
    predictions = model.predict_proba(values[features])[:, 1]
    expected_below_thresh = np.mean(predictions) < thresh

    ind = np.array(individual)
    cost = np.dot(ind, costs)
    res = cost + lbd * expected_below_thresh
    return res,


def evaluate_meaningful(y_name, gamma, scm, obs, features, costs, lbd, r_type, individual):
    # WARNING: for individualized recourse we expect the scm to be abducted already

    intv_dict = indvd_to_intrv(features, individual, obs)
    scm_ = scm.copy()

    # for subpopulation-based recourse at this point nondescendants are fixed
    if r_type == 'subpopulation':
        acs = scm_.dag.get_ancestors_node(y_name)
        intv_dict_causes = {k : intv_dict[k] for k in acs & intv_dict.keys()}
        if len(intv_dict_causes.keys()) != len(intv_dict.keys()):
            logger.debug('Intervention dict contained interventions on non-ascendants of Y ({})'.format(y_name))
        scm_ = scm_.fix_nondescendants(intv_dict_causes, obs)
        scm_.sample_context(scm.get_sample_size())

    # sample from intervened distribution for obs_sub
    values = scm_.compute(do=intv_dict)
    perc_positive = values[y_name].mean()
    meaningfulness_cost = max(perc_positive - gamma, 0)

    ind = np.array(individual)
    cost = np.dot(ind, costs)
    res = cost + lbd * meaningfulness_cost
    return res,


def individualized_post_recourse_predict(scm, obs_pre, obs_post, intv_dict, y_name):
    """
    TODO implement
    """
    scm_ = scm.abduct(obs_pre)
    scm_int = scm.do(intv_dict)
    obs_post_ = obs_post.copy()
    ys = [0, 1]
    for y in ys:
        obs_post_[y_name] = y
        scm_int_abd = scm_int.abd(obs_post_)

        # extract abducted values u' as dictionary

        # compute their joint probability as specified by scm_

        # compute p(y|x_pre)

        # take the product p(u=u')p(y|x_pre)

    # divide the term for y = 1 by the sum of both

    # return result
    pass

def recourse(scm_, features, obs, costs, r_type, t_type, model=None, y_name=None, cleanup=True,
             gamma=None, eta=None, thresh=None, lbd=1.0):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    IND_SIZE = len(features)
    CX_PROB = 0.2
    MX_PROB = 0.5
    NGEN = 100

    toolbox = base.Toolbox()
    toolbox.register("intervene", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.intervene, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=CX_PROB)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MX_PROB)
    toolbox.register("select", tools.selNSGA2)

    if t_type == 'acceptance':
        assert not model is None
        assert not thresh is None
        toolbox.register("evaluate", evaluate, model, thresh, scm_, obs, features, costs, lbd, r_type)
    elif t_type == 'improvement':
        assert not y_name is None
        assert not gamma is None
        toolbox.register("evaluate", evaluate_meaningful, y_name, gamma, scm_, obs, features, costs, lbd, r_type)
    else:
        raise NotImplementedError('only t_types acceptance or improvement are available')

    stats = tools.Statistics(key=lambda ind: np.array(ind.fitness.values))
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    pop, logbook = eaMuPlusLambda(pop, toolbox, 4, 20, CX_PROB, MX_PROB, NGEN, stats=stats, halloffame=hof,
                                  verbose=False)

    winner = list(hof)[0]

    if cleanup:
        del creator.FitnessMin
        del creator.Individual

    return winner, pop, logbook


def recourse_population(scm, model, X, y, U, y_name, costs, proportion=0.5, nsamples=10 ** 2,
                        r_type='individualized', t_type='acceptance',
                        gamma=None, thresh=None, lbd=1.0):
    predictions = model.predict(X)
    ixs_rejected = np.arange(len(predictions))[predictions == 0]
    ixs_recourse = np.random.choice(ixs_rejected, size=math.floor(proportion * len(ixs_rejected)))

    X_new = X.copy()
    y_new = None
    if y is not None:
        y_new = y.copy()
    interventions = []

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
        winner, pop, logbook = recourse(scm_, X.columns, obs, costs, r_type, t_type, model=model, y_name=y_name,
                                        gamma=gamma, thresh=thresh, lbd=lbd)
        intervention = indvd_to_intrv(X.columns, winner, obs)

        interventions.append(winner)

        # compute the actual outcome for this observation
        scm_true = scm.copy()
        scm_true.set_noise_values(U.iloc[ix, :].to_dict())
        scm_true.sample_context(size=1)
        sample = scm_true.compute(do=intervention)
        X_new.iloc[ix, :] = sample[X.columns].to_numpy()
        y_new.iloc[ix] = sample[y_name]

    return ixs_recourse, interventions, X_new, y_new
