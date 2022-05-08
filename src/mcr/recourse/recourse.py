import random
import numpy as np
import logging

from deap import base, creator
from deap.algorithms import eaMuPlusLambda
from deap import tools

from mcr.recourse.evaluation import GreedyEvaluator, similar

# LOGGING

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# RECOURSE FUNCTIONS

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
        goal_met = gamma < goal_cost
    elif not eta is None:
        goal_met = eta < goal_cost

    if not goal_met:
        winner = [0 for _ in winner]
        goal_cost, intv_cost = eval_cost(winner)

    # cleanup
    if cleanup:
        del creator.FitnessMin
        del creator.Individual
        del evaluator

    return winner, pop, logbook, goal_cost, intv_cost
