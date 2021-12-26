import random
import math
import pandas as pd
import numpy as np
import torch
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from rfi.backend.causality.scm import BinomialBinarySCM
from rfi.backend.causality.dags import DirectedAcyclicGraph

from deap import base, creator
from deap.algorithms import eaMuPlusLambda
from deap import tools
from tqdm import tqdm



# RECOURSE FUNCTIONS

def indvd_to_intrv(features, individual, obs):
    dict = {}
    for ii in range(len(features)):
        if individual[ii]:
            dict[features[ii]] = (obs[features[ii]] + individual[ii]) % 2
    return dict


def evaluate(model, scm, obs, features, individual):
    intv_dict = indvd_to_intrv(features, individual, obs)

    # sample from intervened distribution for obs_sub
    values = scm.compute(do=intv_dict)
    predictions = model.predict_proba(values[features])[:, 1]
    expected_below_thresh = np.mean(predictions) < 0.5

    ind = np.array(individual)
    cost = np.dot(ind, costs)
    return cost + expected_below_thresh


def evaluate_meaningful(scm_abd, features, individual):
    intv_dict = indvd_to_intrv(features, individual, obs)
    # sample from intervened distribution for obs_sub
    values = scm_abd.compute(do=intv_dict)
    return values[y_name].mean(), values[y_name].std()


def recourse(model, scm_abd, features, obs):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    IND_SIZE = len(X.columns)
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
    toolbox.register("evaluate", evaluate, model, scm_abd, obs, features)

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


def recourse_population(scm, model, X, y, U, proportion=0.5, nsamples=10 ** 2):
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
        # TODO incorporate condition about whether to sample for the individual or subpopulation-based
        # for subpopulation-based use do operator to fix the values for all non-descendants of Y
        scm_abd = scm.abduct(obs, n_samples=nsamples)
        cntxt = scm_abd.sample_context(size=nsamples)

        # compute optimal action
        winner, pop, logbook = recourse(model, scm_abd, X.columns, obs)
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


# LOGGING

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# DEFINE DATA GENERATING MECHANISM

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.3)
sigma_low = torch.tensor(0.2)

scm = BinomialBinarySCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 0]]),
        var_names=['vaccinated', 'covid-free', 'symptom-free']
    ),
    p_dict={'vaccinated': sigma_high,
            'symptom-free': sigma_low, 'covid-free': sigma_medium}
)

costs = np.array([0.5, 0.1])
y_name = 'covid-free'


# GENERATE THREE BATCHES OF DATA

N = 10 ** 3

noise = scm.sample_context(N)
df = scm.compute()

X = df[df.columns[df.columns != y_name]]
y = df[y_name]

batch_size = math.floor(N / 3)

batches = []
i = 0

while i < N:
    X_i, y_i = X.iloc[i:i + batch_size, :], y.iloc[i:i + batch_size]
    U_i = noise.iloc[i:i + batch_size, :]
    batches.append((X_i, y_i, U_i))
    i += batch_size


# FITTING MODEL 1 ON BATCH 0

model = LogisticRegression()
model.fit(batches[0][0], batches[0][1])

model.predict(batches[1][0])

perf1 = log_loss(batches[0][1], model.predict_proba(batches[0][0]))
perf2 = accuracy_score(batches[0][1], model.predict(batches[0][0]))


# TESTING RECOURSE ON ONE OBSERVATION

obs = batches[0][0].iloc[5, :]
scm_abd = scm.abduct(obs)
scm_abd.sample_context(1000)
winner, log, population = recourse(model, scm_abd, batches[0][0].columns, obs)


# APPLYING RECOURSE TO WHOLE POPULATION

## Application to batch 1
## assessing how well the model predicts in that environment
## assessing meaningfulness of the recourse

batches_post = []
batches_post.append(batches[0])

ixss_recourse = [[]]
interventionss = [[]]

for ii in [1, 2]:
    ixs_recourse, interventions, X_new, y_new = recourse_population(scm, model, batches[ii][0], batches[ii][1],
                                                                    batches[ii][2], proportion=1.0)
    batches_post.append((X_new, y_new))
    ixss_recourse.append(ixs_recourse)
    interventionss.append(interventions)


# how successful is recourse on the original model?

predict_post = model.predict(batches_post[1][0])
predict_pre = model.predict(batches[1][0])
df_rec1 = pd.DataFrame()
df_rec1['y_hat_post'] = predict_post[ixss_recourse[1]]
df_rec1['y_hat_pre'] = predict_pre[ixss_recourse[1]]
invs = np.array(interventionss[1])
df_rec1['y_pre'] = np.array(batches[1][1].iloc[ixss_recourse[1]])
df_rec1['y_post'] = np.array(batches_post[1][1].iloc[ixss_recourse[1]])
ii = 0
for column in X.columns:
    df_rec1['pre' + column] = np.array(batches[1][0][column].iloc[ixss_recourse[1]])
    df_rec1['int ' + column] = invs[:, ii]
    df_rec1['post' + column] = np.array(batches_post[1][0][column].iloc[ixss_recourse[1]])
    ii += 1

strategies = df_rec1[['presymptom-free', 'prevaccinated', 'int symptom-free', 'int vaccinated']].drop_duplicates()

print('Recourse strategy:')
print(strategies)

accuracy_pre = accuracy_score(df_rec1['y_hat_pre'], df_rec1['y_pre'])
accuracy_post = accuracy_score(df_rec1['y_hat_post'], df_rec1['y_post'])

print('Accuracy on recourse seeking pre-recourse: {}'.format(accuracy_pre))
print('Accuracy on recourse seeking post-recourse: {}'.format(accuracy_post))

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
predict_all_comb = model.predict(data)
print('Feature names: {}'.format(X.columns))
for ii in range(len(data)):
    print('prediction for {}: {}'.format(data[ii], predict_all_comb[ii]))


## Application to batch 2
## refitting model on batch 0 and 1
## computing recourse on batch 2 with respect to original model
## assessing whether recourse is honored by the new model

# fit model on batches 0 and 1
X_train2 = pd.concat([batches_post[0][0], batches_post[1][0]])
y_train2 = pd.concat([batches_post[0][1], batches_post[1][1]])

model2 = LogisticRegression()
model2.fit(X_train2, y_train2)
predict2 = model2.predict(batches_post[2][0])

perf1 = log_loss(batches_post[2][1], model.predict_proba(batches[2][0]))
perf2 = accuracy_score(batches_post[2][1], model.predict(batches[2][0]))

predict2[np.array(ixss_recourse[2], dtype=np.int)]

# predict on batch 2 and see whether recourse is honored

df_rec = pd.DataFrame([])
invs = np.array(interventionss[2])
df_rec['int ' + X.columns[0]] = invs[:, 0]
df_rec['int ' + X.columns[1]] = invs[:, 1]
df_rec['y_new'] = np.array(batches_post[2][1].iloc[ixss_recourse[2]])
df_rec['pred'] = predict2[np.array(ixss_recourse[2], dtype=np.int)]
for column in X.columns:
    df_rec[column] = np.array(batches_post[2][0][column].iloc[ixs_recourse])

percentage_honored = df_rec['pred'].mean()
print('Recourse honored only for {} per cent'.format(percentage_honored))
