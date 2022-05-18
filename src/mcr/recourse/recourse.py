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

def project_within_bounds(proposal, bound):
    if proposal < bound[0]:
        return bound[0]
    elif proposal > bound[1]:
        return bound[0]
    else:
        return proposal

def initrepeat_mixed(container, bounds, X_init=None, obs=None, n_digits=None):
    # tuple of the form (2, 5, inf) for binary, categorical, continuous data
    ind = []
    for jj in range(len(bounds)):
        if X_init is None or obs is None:
            if isinstance(bounds[jj][0], int) and isinstance(bounds[jj][1], int):
                ind.append(random.randint(bounds[jj][0], bounds[jj][1]))
            else:
                proposal = random.random()
                proposal = project_within_bounds(proposal, bounds[jj])
                ind.append(proposal)
        else:
            proposal_sample = X_init.sample(1)
            proposal = proposal_sample - obs
            ind = list(proposal.values[0])

    if n_digits is not None:
        ind = [round(x, n_digits) for x in ind]

    return container(ind)

def mutate_mixed(individual, indpb, mu, sigma, bounds, n_digits):
    for jj in range(len(individual)):
        if random.random() < indpb:
            if isinstance(bounds[jj][0], int) and isinstance(bounds[jj][1], int):
                if individual[jj] == bounds[jj][1]:
                    individual[jj] -= 1
                elif individual[jj] == bounds[jj][0]:
                    individual[jj] += 1
                else:
                    individual[jj] = individual[jj] + random.choice([-1, 1])
            else:
                proposal = tools.mutGaussian([individual[jj]], mu, sigma, indpb)[0][0]
                individual[jj] = project_within_bounds(proposal, bounds[jj])

    if n_digits is not None:
        for ii in range(len(individual)):
            individual[ii] = round(individual[ii], n_digits)

    return individual,


# RECOURSE FUNCTIONS

def recourse(scm_, features, obs, costs, r_type, t_type, predict_log_proba=None, y_name=None, cleanup=True, gamma=None,
             eta=None, thresh=None, lbd=1.0, subpopulation_size=500, NGEN=400, CX_PROB=0.3, MX_PROB=0.05,
             POP_SIZE=1000, rounding_digits=2, binary=False, multi_objective=False, return_stats=False, X=None):

    evaluator = GreedyEvaluator(scm_, obs, costs, features, lbd, rounding_digits=rounding_digits,
                                subpopulation_size=subpopulation_size, predict_log_proba=predict_log_proba,
                                y_name=y_name, multi_objective=multi_objective)

    bounds = []
    for node in features:
        scm_bounds = scm_.bounds[node]
        tupl = (scm_bounds[0] - obs[node], scm_bounds[1] - obs[node])
        if isinstance(scm_bounds[0], int) and isinstance(scm_bounds[1], int):
            tupl = (int(tupl[0]), int(tupl[1]))
        bounds.append(tupl)

    if y_name is None:
        y_name = scm_.predict_target

    if multi_objective:
        raise NotImplementedError('Not implemented yet.')
        creator.create("FitnessMin", base.Fitness, weights=(lbd, -1.0))
    else:
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    IND_SIZE = len(features)

    toolbox = base.Toolbox()
    # if binary:
    #     toolbox.register("intervene", random.randint, 0, 1)
    # else:
    #     toolbox.register("intervene", random.random)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.intervene, n=IND_SIZE)
    toolbox.register("individual", initrepeat_mixed, creator.Individual, bounds, X[features],
                     obs[features], rounding_digits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=CX_PROB)
    # if binary:
    #     toolbox.register("mutate", tools.mutFlipBit, indpb=MX_PROB)
    # else:
    #     toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=MX_PROB)
    toolbox.register("mutate", mutate_mixed, indpb=MX_PROB, mu=0, sigma=1, bounds=bounds, n_digits=rounding_digits)
    toolbox.register("select", tools.selNSGA2)

    if t_type == 'acceptance':
        assert not predict_log_proba is None
        assert not thresh is None
        toolbox.register("evaluate", evaluator.evaluate, eta, thresh, r_type)
    elif t_type == 'improvement':
        assert not y_name is None
        assert not gamma is None
        toolbox.register("evaluate", evaluator.evaluate_meaningful, gamma, r_type)
    elif t_type == 'counterfactual':
        assert not predict_log_proba is None
        toolbox.register("evaluate", evaluator.evaluate_ci, thresh)
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

    invds = np.array(list(hof))
    perf = np.array([x.values for x in list(hof.keys)])
    if multi_objective:
        # TODO fix. indvs and perf do no thave the same order
        raise RuntimeError('Your code should not end up here.')
        min_cost_constrained = np.min(perf[perf[:, 0] > 0.95, 1])
        best_ix = np.where(perf[:, 1] == min_cost_constrained)[0][0]
        winner = invds[best_ix, :]
    else:
        winner = invds[0, :]
        best_perf = 0
        for ii in np.arange(len(invds)):
            perf_tmp = toolbox.evaluate(invds[ii, :])[0]
            if perf_tmp > best_perf:
                best_perf = perf_tmp
                winner = invds[ii, :]

    winner = [round(x, ndigits=rounding_digits) for x in winner]

    def eval_cost(winner):
        if t_type == 'acceptance':
            goal_cost, intv_cost = evaluator.evaluate(eta, thresh, r_type, winner, return_split=True)
        elif t_type == 'improvement':
            goal_cost, intv_cost = evaluator.evaluate_meaningful(gamma, r_type, winner, return_split=True)
        elif t_type == 'counterfactual':
            goal_cost, intv_cost = evaluator.evaluate_ci(thresh, winner, return_split=True)
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
        evaluator.clear_cache()
        toolbox.unregister('evaluate')
        toolbox.unregister('mutate')
        toolbox.unregister('select')
        toolbox.unregister('individual')
        toolbox.unregister('mate')
        toolbox.unregister('population')
        del evaluator
        if not return_stats:
            if not multi_objective:
                hof.clear()
            del pop
            del logbook

    if return_stats:
        return winner, pop, hof, logbook, goal_cost, intv_cost
    else:
        return winner, goal_cost, intv_cost
