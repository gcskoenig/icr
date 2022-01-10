import logging
import copy
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from mar.causality import BinomialBinarySCM
from mar.causality import DirectedAcyclicGraph
from mar.recourse import recourse_population, save_recourse_result

import numpy as np
import math
import torch
import pandas as pd

# DEFINE LOGGING LEVEL

logging.getLogger().setLevel(logging.INFO)

savepath = "../experiments/robustness/"

gamma = 0.7
eta = 0.7
lbd = 4

# DEFINE DATA GENERATING MECHANISM

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.01)

scm = BinomialBinarySCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]]),
        var_names=['vaccinated', 'contacts', 'mask', 'covid-free', 'symptom-free']
    ),
    p_dict={'vaccinated': sigma_high, 'contacts': sigma_high, 'mask': sigma_high,
            'symptom-free': sigma_low, 'covid-free': sigma_medium}
)

costs = np.array([1.0, 1.0, 1.0, 0.1])
y_name = 'covid-free'


# GENERATE THREE BATCHES OF DATA

N = 10**4 * 3

noise = scm.sample_context(N)
df = scm.compute()

X = df[df.columns[df.columns != y_name]]
y = df[y_name]

batch_size = math.floor(N / 3)

logging.info('Creating three batches of data with {} observations'.format(batch_size))

batches = []
i = 0

while i < N:
    X_i, y_i = X.iloc[i:i + batch_size, :], y.iloc[i:i + batch_size]
    U_i = noise.iloc[i:i + batch_size, :]
    batches.append((X_i, y_i, U_i))
    i += batch_size

# FITTING MODEL 1 ON BATCH 0

logging.info('Fitting a model on batch 0')

model = LogisticRegression()
model.fit(batches[0][0], batches[0][1])

model.predict(batches[1][0])

perf1 = log_loss(batches[0][1], model.predict_proba(batches[0][0]))
perf2 = accuracy_score(batches[0][1], model.predict(batches[0][0]))


# EXPERIMENTS

r_types = ['subpopulation', 'individualized']
t_types = ['improvement', 'acceptance']


## How well does the model predict in different pre- and post-recourse enviornments?
## How meaningful is the recourse? (i.e. what is the improvement probability)
## What is the acceptance probability?

# t_type = 'improvement'
# r_type = 'individualized'

for r_type in r_types:
    for t_type in t_types:
        savepath_run = savepath + '{}_{}/'.format(t_type, r_type)

        try:
            os.mkdir(savepath_run)
        except FileExistsError as err:
            logging.info('Folder {} already exists'.format(savepath_run))
        except Exception as err:
            print('could not create folder.')
            raise err

        logging.info('Recourse type: {}, {}'.format(r_type, t_type))

        ## Perform CR on batches 1 and 2

        logging.info('Batches 1 and 2 are replace with post-recourse data')
        logging.info('Batch 0 is left as-is')

        # for batches 1 and 2 recourse is performed
        result_tuples = []

        for ii in [1, 2]:
            tpl = recourse_population(scm, batches[ii][0], batches[ii][1],
                                      batches[ii][2], y_name, costs,
                                      proportion=0.8, r_type=r_type,
                                      t_type=t_type, gamma=gamma, eta=eta,
                                      thresh=0.5, lbd=lbd,
                                      subpopulation_size=100,
                                      model=model,
                                      use_scm_pred=False)
            result_tuples.append(tpl)
            X_pre, y_pre, y_hat_pre, interventions, X_post, y_post, h_post, costss, stats = tpl
            logging.info(stats)

        save_recourse_result(savepath_run + 'batch1_', result_tuples[0])
        save_recourse_result(savepath_run + 'batch2_', result_tuples[1])

        ## Application to batch 2
        ## refitting model on batch 0 and 1
        ## computing recourse on batch 2 with respect to original model
        ## assessing whether recourse is honored by the new model

        logging.info('A second model is fit on batches 0 (pre-recourse) and 1 (post-recourse)')

        batches_post = copy.deepcopy(batches)

        for ii in [1, 2]:
            X_post, y_post = result_tuples[ii - 1][4], result_tuples[ii - 1][5]
            batches_post[ii][0].iloc[X_post.index, :] = X_post
            batches_post[ii][1].iloc[y_post.index] = y_post

        # fit model on batches 0 and 1
        # batch 0 post is identical to batch 0 pre
        X_train2 = batches_post[1][0]
        y_train2 = batches_post[1][1]

        model2 = LogisticRegression()
        model2.fit(X_train2, y_train2)

        logging.info('The refit on pre- and post-recourse data has coefficients {}'.format(model2.coef_))

        X_post_2, y_post_2 = result_tuples[1][4], result_tuples[1][5]
        invs = result_tuples[1][3]
        recourse_performed = invs[invs.sum(axis=1) >= 1].index
        X_post_2 = X_post_2.loc[recourse_performed, :]
        y_post_2 = y_post_2.loc[recourse_performed]

        if len(recourse_performed) > 0:
            predict2 = model2.predict(X_post_2)
            np.save(savepath_run + 'predict2.npy', predict2)

            perf1 = log_loss(y_post_2, model.predict_proba(X_post_2), labels=[0, 1])
            perf2 = accuracy_score(y_post_2, model.predict(X_post_2))

            logging.info("Performance of refit on post-recourse data: {} log-loss and {}% accuracy".format(perf1, perf2))

            # predict on batch 2 and see whether recourse is honored

            percentage_honored = np.mean(predict2)
            logging.info('Recourse honored only for {} per cent'.format (percentage_honored))