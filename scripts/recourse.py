import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from mar.causality import BinomialBinarySCM
from mar.causality import DirectedAcyclicGraph
from mar.recourse import recourse, recourse_population

import numpy as np
import math
import torch
import pandas as pd

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
# a individualized

obs = batches[0][0].iloc[5, :]
scm_abd = scm.abduct(obs)
scm_abd.sample_context(1000)
winner, log, population = recourse(model, scm_abd, batches[0][0].columns, obs, costs)

# b subpopulation-based

obs = batches[0][0].iloc[5, :]
scm_ = scm.do(obs[scm.dag.get_nondescendants(y_name)])
scm_.sample_context(1000)
winner, log, population = recourse(model, scm_, batches[0][0].columns, obs, costs)


# APPLYING RECOURSE TO WHOLE POPULATION

## How well does the model predict in different pre- and post-recourse enviornments?
## How meaningful is the recourse? (i.e. what is the improvement probability)
## What is the acceptance probability?

r_type = 'individualized'

logging.info('Recourse type: {}'.format(r_type))

## Perform CR on batches 1 and 2

logging.info('Recourse is computed for batches 1 and 2')

batches_post = []

# batch 0 is left as is
batches_post.append(batches[0])

# for batches 1 and 2 recourse is performed
ixss_recourse = [[]]
interventionss = [[]]

for ii in [1, 2]:
    ixs_recourse, interventions, X_new, y_new = recourse_population(scm, model, batches[ii][0], batches[ii][1],
                                                                    batches[ii][2], y_name, costs, proportion=1.0,
                                                                    r_type=r_type)
    batches_post.append((X_new, y_new))
    ixss_recourse.append(ixs_recourse)
    interventionss.append(interventions)


# acceptance probability on model 0

logging.info('The models predictions on pre- and post-recourse batch 1 are computed')

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

logging.info('The recourse strategies for all possible input combinations are computed and printed:')
strategies = df_rec1[['presymptom-free', 'prevaccinated', 'int symptom-free', 'int vaccinated']].drop_duplicates()

print('Recourse strategy:')
print(strategies)

logging.info('Pre- and post-recourse accuracies for batch 1 are computed.')

accuracy_pre = accuracy_score(df_rec1['y_hat_pre'], df_rec1['y_pre'])
accuracy_post = accuracy_score(df_rec1['y_hat_post'], df_rec1['y_post'])

print('Accuracy on recourse seeking pre-recourse: {}'.format(accuracy_pre))
print('Accuracy on recourse seeking post-recourse: {}'.format(accuracy_post))


logging.info('Compute all possible prediction strategies for model 0.')

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
predict_all_comb = model.predict(data)
print('Feature names: {}'.format(X.columns))
for ii in range(len(data)):
    print('prediction for {}: {}'.format(data[ii], predict_all_comb[ii]))


## Application to batch 2
## refitting model on batch 0 and 1
## computing recourse on batch 2 with respect to original model
## assessing whether recourse is honored by the new model

logging.info('A second model is fit on batches 0 (pre-recourse) and 1 (post-recourse)')

# fit model on batches 0 and 1
# batch 0 post is identical to batch 0 pre
X_train2 = pd.concat([batches_post[0][0], batches_post[1][0]])
y_train2 = pd.concat([batches_post[0][1], batches_post[1][1]])

model2 = LogisticRegression()
model2.fit(X_train2, y_train2)
predict2 = model2.predict(batches_post[2][0])

perf1 = log_loss(batches_post[2][1], model.predict_proba(batches[2][0]))
perf2 = accuracy_score(batches_post[2][1], model.predict(batches[2][0]))

predict2[np.array(ixss_recourse[2], dtype=np.int)]

# predict on batch 2 and see whether recourse is honored
logging.info('Compute recourse strategies with respect to original model.')
logging.info('Assess whether recourse honored by refitted model.')

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
