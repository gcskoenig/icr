import logging
logging.getLogger().setLevel(logging.INFO)

import os
import copy
import random
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import math

from mar.recourse import recourse_population, save_recourse_result
from mar.causality.scm import BinomialBinarySCM

def load_problem(path):
    scm = BinomialBinarySCM.load(path)
    y_name = scm.predict_target
    return scm, y_name


def run_robustness_experiment(savepath, scm, y_name, gamma, eta, lbd, thresh, costs, N):

    # GENERATE THREE BATCHES OF DATA

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

    r_types = ['individualized', 'improvement']
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
                                          lbd=lbd,
                                          thresh=thresh,
                                          subpopulation_size=200,
                                          model=model,
                                          use_scm_pred=False)
                result_tuples.append(tpl)
                U, X_pre, y_pre, y_hat_pre, interventions, X_post, y_post, h_post, costss, stats = tpl
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
                X_post, y_post = result_tuples[ii - 1][5], result_tuples[ii - 1][6]
                batches_post[ii][0].iloc[X_post.index, :] = X_post
                batches_post[ii][1].iloc[y_post.index] = y_post


            # fit model on batch 1
            # batch 0 post is identical to batch 0 pre
            X_train2 = batches_post[1][0]
            y_train2 = batches_post[1][1]

            model2 = LogisticRegression()
            model2.fit(X_train2, y_train2)

            logging.info('The refit on pre- and post-recourse data has coefficients {}'.format(model2.coef_))

            models = [model, model2]

            for nr in [0, 1]:
                np.savetxt(savepath_run + 'model{}_coef.csv'.format(nr), np.array(models[nr].coef_), delimiter=',')
                X_post_nr, y_post_nr = result_tuples[nr][5], result_tuples[nr][6]
                invs = result_tuples[nr][4]
                recourse_performed = invs[invs.sum(axis=1) >= 1].index
                X_post_nr = X_post_nr.loc[recourse_performed, :]
                y_post_nr = y_post_nr.loc[recourse_performed]

                if len(recourse_performed) > 0:
                    predict2 = models[nr].predict(X_post_nr)
                    np.savetxt(savepath_run + 'predict{}.csv'.format(nr), predict2, delimiter=',')

                    perf1 = log_loss(y_post_nr, model.predict_proba(X_post_nr), labels=[0, 1])
                    perf2 = accuracy_score(y_post_nr, model.predict(X_post_nr))

                    logging.info("Performance of refit on post-recourse data: {} log-loss and {}% accuracy".format(perf1, perf2))

                    # predict on batch 2 and see whether recourse is honored

                    percentage_honored = np.mean(predict2)
                    logging.info('-----')
                    logging.info('Recourse honored only for {} per cent'.format(percentage_honored))
                    logging.info('=====')

if __name__ == '__main__':
    # DEFINE LOGGING LEVEL

    # logging.getLogger().setLevel(logging.INFO)
    # parsing command line arguments
    parser = argparse.ArgumentParser("Create recourse experiments. " +
                                     "For every configuration a separate folder is created. " +
                                     "Within every folder a folder for every interation is created." +
                                     "The savepath specifies the folder in which these folders shall be placed.")

    parser.add_argument("scm_loadpath", help="loadpath for scm to be used", default=None, type=str)
    parser.add_argument("gamma", help="gammas for recourse", type=float)
    parser.add_argument("lbd", help="lambda for optimization", default=10.0, type=float)
    parser.add_argument("thresh", help="threshs for prediction and recourse", type=float)
    parser.add_argument("N", help="Number of observations", type=int)
    parser.add_argument("savepath",
                        help="savepath for the experiment folder. either relative to working directory or absolute.",
                        type=str)

    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--logging_level", help="logging-level", default=20, type=int)

    args = parser.parse_args()

    # set logging settings
    logging.getLogger().setLevel(args.logging_level)
    random.seed(args.seed)

    scm, y_name = load_problem(args.scm_loadpath)
    costs = np.load(args.scm_loadpath + 'costs.npy')

    # expects that we are in a directory with a subfolder called "experiments"
    # relative save paths
    config_id = random.randint(0, 1024)
    savepath_config = args.savepath + 'gamma_{}_N_{}_id_{}/'.format(args.gamma, args.N, config_id)

    n_tries = 0
    done = False
    while n_tries < 5 and not done:
        try:
            n_tries += 1
            os.mkdir(savepath_config)
            done = True
        except Exception as err:
            logging.warning('Could not generate folder...{}'.format(savepath_config))

    run_robustness_experiment(savepath_config, scm, y_name, args.gamma, args.gamma, args.lbd, args.thresh, costs, args.N)
