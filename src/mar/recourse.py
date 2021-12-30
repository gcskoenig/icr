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


def evaluate(model, scm, obs, features, costs, individual):
    intv_dict = indvd_to_intrv(features, individual, obs)

    # sample from intervened distribution for obs_sub
    values = scm.compute(do=intv_dict)
    predictions = model.predict_proba(values[features])[:, 1]
    expected_below_thresh = np.mean(predictions) < 0.5

    ind = np.array(individual)
    cost = np.dot(ind, costs)
    res = cost + expected_below_thresh
    return res,

def evaluate_meaningful(scm_abd, features, y_name, individual, obs):
    intv_dict = indvd_to_intrv(features, individual, obs)
    # sample from intervened distribution for obs_sub
    values = scm_abd.compute(do=intv_dict)
    return values[y_name].mean(), values[y_name].std()


def recourse(model, scm_, features, obs, costs):
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
    toolbox.register("evaluate", evaluate, model, scm_, obs, features, costs)

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
    return winner, pop, logbook


def recourse_population(scm, model, X, y, U, y_name, costs, proportion=0.5, nsamples=10 ** 2, r_type='individualized'):
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

        if r_type == 'subpopulation':
            nds = scm.dag.get_nondescendants(y_name)
            scm_ = scm.do(obs[nds])
        elif r_type == 'individualized':
            scm_ = scm.abduct(obs, n_samples=nsamples)
        else:
            raise NotImplementedError('r_type must be in {}'.format(['individualized', 'subpopulation']))

        # compute optimal action
        cntxt = scm_.sample_context(size=nsamples)
        winner, pop, logbook = recourse(model, scm_, X.columns, obs, costs)
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
