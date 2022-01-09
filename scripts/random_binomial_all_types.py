"""
The goal of the script is to compare the results of the four different recourse methods that are discussed in
the paper on meaningful algorithmic recourse.

In this script we randomly generate a scm/prediction problem.
We sample two batches of data, of which we use one to fit a ML model.
On the second batch we apply the four types of recourse:

- algorithmic recourse (Karimi et al.), both individualized and subpopulation-based
- meaningful algorithmic recourse (our suggestion), both individualized and subpopulation-based

The following results are saved:

- problem_setup.json (the parameters that generated the scm)
- the generated scm which includes
    - the dag as adjacency_matrix
    - the probability dictionary
- the generated data
- the model's coefficients

And for every recourse type we save:
- the pre- and post-recourse dataframes for recourse seeking individuals as well as pre- and post-recourse predictions
- the experiment result statistics (stat.json) that includes the recourse hyperparemeters

All data is saved within one folder, which is given a randomly assigned id
"""

import json
import logging
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
import random
import os
import argparse

from mar.causality.dags import DirectedAcyclicGraph
from mar.causality.scm import BinomialBinarySCM
from mar.recourse import recourse_population


logging.getLogger().setLevel(20)

# script functions

def generate_problem(N, p, min_in_degree, max_out_degree, max_uncertainty, seed=42):
    random.seed(seed)
    scm_return = None
    y_name = None
    while y_name is None:
        dag = DirectedAcyclicGraph.random_dag(N, p=p)
        scm = BinomialBinarySCM(dag)
        y_name = None

        for node in scm.model.keys():
            thresh = random.uniform(0, max_uncertainty)
            below_thresh = scm.model[node]['noise_distribution'].probs.item() <= thresh
            has_min_in_degree = len(scm.model[node]['parents']) >= min_in_degree
            has_max_out_degree = len(scm.model[node]['children']) <= max_out_degree
            degree_ok = has_min_in_degree and has_max_out_degree
            if below_thresh and degree_ok:
                y_name = node
                scm_return = scm
                logging.info('Found target variable: {}'.format(y_name))
                logging.info('y_name: {}'.format(scm.model[node]))
                break
    scm_return.set_prediction_target(y_name)
    return scm_return, y_name


def run_experiment(N_nodes, p, max_uncertainty, min_in_degree, max_out_degree, seed, N,
                   lbd, gamma, thresh, savepath, use_scm_pred=False, iterations=5):
    try:
        os.mkdir(savepath)
    except OSError as err:
        print(err)
        logging.warning('Creating of directory %s failed' % savepath)
    else:
        logging.info('Creation of directory %s successful' % savepath)

    logging.info('generating problem...')
    # generate problem
    scm, y_name = generate_problem(N_nodes, p, min_in_degree, max_out_degree, max_uncertainty, seed=seed)
    noise = scm.sample_context(N)
    df = scm.compute()
    X = df[df.columns[df.columns != y_name]]
    y = df[y_name]

    costs = np.random.uniform(0, lbd / (N_nodes - 1), N_nodes - 1)

    # split into batches
    n_batches = 2
    batch_size = math.floor(N / n_batches)
    batches = []
    i = 0

    while i < N:
        X_i, y_i = X.iloc[i:i + batch_size, :], y.iloc[i:i + batch_size]
        U_i = noise.iloc[i:i + batch_size, :]
        batches.append((X_i, y_i, U_i))
        i += batch_size

    logging.info('Split the data into {} batches of ~{} elements'.format(n_batches, batch_size))


    # fitting standard logistic regression on the first batch

    logging.info('Fitting model...')
    model = LogisticRegression()
    model.fit(batches[0][0], batches[0][1])


    # CHECKPOINT: SAVE ALL RELEVANT DATA

    problem_setup = {'N': N, 'N_nodes': N_nodes, 'p': p, 'max_uncertainty': max_uncertainty,
                     'min_in_degree': min_in_degree,
                     'max_out_degree': max_out_degree, 'seed': seed,
                     'use_scm_pred': use_scm_pred}

    logging.info('Storing all relevant data...')
    # problem setup
    with open(savepath + 'problem_setup.json', 'w') as f:
        json.dump(problem_setup, f)
    # model coefficients
    np.save(savepath + 'model_coef.npy', np.array(model.coef_))
    # scm
    scm.save(savepath + 'scm')
    # data
    batches[0][0].to_csv(savepath + 'X_train.csv')
    batches[0][1].to_csv(savepath + 'y_train.csv')
    batches[1][0].to_csv(savepath + 'X_test.csv')
    batches[1][1].to_csv(savepath + 'y_test.csv')


    # run all types of recourse on the setting
    logging.info('Run all types of recourse...')

    r_types = ['subpopulation', 'individualized']
    t_types = ['acceptance', 'improvement']

    all_combinations = []
    for r_type in r_types:
        for t_type in t_types:
            all_combinations.append((r_type, t_type))



    for r_type, t_type in all_combinations:
        logging.info("combination: {} {}".format(r_type, t_type))
        savepath_config_type = savepath_config + '{}-{}/'.format(t_type, r_type)
        os.mkdir(savepath_config_type)

        for ii in range(iterations):
            it_path = savepath_config_type + '{}/'.format(ii)
            os.mkdir(it_path)

            # perform recourse on subpopulation
            X_pre, y_pre, y_hat_pre, invs, X_post, y_post, h_post, costss, stats = recourse_population(scm, batches[1][0],
                                                                                                       batches[1][1],
                                                                                                       noise,
                                                                                                       y_name, costs,
                                                                                                       proportion=1.0,
                                                                                                       r_type=r_type,
                                                                                                       t_type=t_type,
                                                                                                       gamma=gamma,
                                                                                                       eta=gamma,
                                                                                                       thresh=thresh,
                                                                                                       lbd=lbd,
                                                                                                       model=model,
                                                                                                       use_scm_pred=use_scm_pred)

            logging.info('Saving results for {}_{}...'.format(t_type, r_type))
            # save results
            savepath_exp = savepath + '{}_{}'.format(r_type, t_type)
            X_pre.to_csv(savepath_exp + '_X_pre.csv')
            y_pre.to_csv(savepath_exp + '_y_pre.csv')
            y_hat_pre.to_csv(savepath_exp + '_y_hat_pre.csv')
            invs.to_csv(savepath_exp + '_invs.csv')
            X_post.to_csv(savepath_exp + '_X_post.csv')
            y_post.to_csv(savepath_exp + '_y_post.csv')
            h_post.to_csv(savepath_exp + '_h_post.csv')
            costss.to_csv(savepath_exp + '_costss.csv')

            try:
                with open(savepath_exp + '_stats.json', 'w') as f:
                    json.dump(stats, f)
            except Exception as exc:
                logging.warning('stats.json could not be saved.')
                logging.info('Exception: {}'.format(exc))

            logging.info('Done.')


# run_experiment(30, 0.8, 0.2, 3, 1000, 42, 2500,
#                10, 0.9, 0.5, '../experiments/random_all_types/')

if __name__ == '__main__':
    # parsing command line arguments
    parser = argparse.ArgumentParser("Create recourse experiments. " +
                                     "For every configuration a separate folder is created. " +
                                     "Within every folder a folder for every interation is created." +
                                     "The savepath specifies the folder in which these folders shall be placed.")

    parser.add_argument("savepath",
                        help="savepath for the experiment folder. either relative to working directory or absolute.",
                        type=str)
    parser.add_argument("N_nodes", help="List with number of nodes to generate", type=int)
    parser.add_argument("N", help="Number of observations", type=int)
    parser.add_argument("gamma", help="gammas for recourse", type=float)
    parser.add_argument("thresh", help="threshs for prediction and recourse", type=float)
    parser.add_argument("n_iterations", help="number of runs per configuration", type=int)

    parser.add_argument("--lbd", help="lambda for optimization", default=10.0, type=float)
    parser.add_argument("--p", help="List with edge probabilities to generate", default=0.8, type=float)
    parser.add_argument("--max_uncertainty", help="Maximum p for y node", default=0.5, type=float)
    parser.add_argument("--min_in_degree", help="minium in-degree for y_node", default=3, type=int)
    parser.add_argument("--max_out_degree", help="maximum out-degree for y_node", default=1000, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)

    parser.add_argument("--logging_level", help="logging-level", default=20, type=int)


    args = parser.parse_args()

    # set logging settings
    logging.getLogger().setLevel(args.logging_level)

    # expects that we are in a directory with a subfolder called "experiments"
    # relative save paths
    config_id = random.randint(0, 1024)
    savepath_config = args.savepath + 'gamma_{}_M_{}_id_{}/'.format(args.gamma, args.N_nodes, config_id)

    n_tries = 0
    done = False
    while n_tries < 5 and not done:
        try:
            n_tries += 1
            os.mkdir(savepath_config)
            done = True
        except Exception as err:
            logging.warning('Could not generate folder...{}'.format(savepath_config))

    run_experiment(args.N_nodes, args.p, args.max_uncertainty, args.min_in_degree, args.max_out_degree,
                   args.seed, args.N, args.lbd, args.gamma, args.thresh, savepath_config,
                   iterations=args.n_iterations, use_scm_pred=False)
