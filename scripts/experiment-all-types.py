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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import math
import random
import os
import argparse

from mcr.causality.dags import DirectedAcyclicGraph
from mcr.causality.scm import BinomialBinarySCM, SigmoidBinarySCM
from mcr.recourse import recourse_population, save_recourse_result


logging.getLogger().setLevel(20)

# script functions

def generate_problem(N, p, min_in_degree, out_degree, max_uncertainty, seed=42):
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
            has_out_degree = len(scm.model[node]['children']) == out_degree
            degree_ok = has_min_in_degree and has_out_degree
            if below_thresh and degree_ok:
                y_name = node
                scm_return = scm
                logging.info('Found target variable: {}'.format(y_name))
                logging.info('y_name: {}'.format(scm.model[node]))
                break
    scm_return.set_prediction_target(y_name)
    return scm_return, y_name


def load_problem(path, type='binomial'):
    scm = None
    if type == 'binomial':
        scm = BinomialBinarySCM.load(path)
    elif type == 'sigmoid':
        scm = SigmoidBinarySCM.load(path)
    y_name = scm.predict_target
    return scm, y_name


def run_experiment(N_nodes, p, max_uncertainty, min_in_degree, out_degree, seed, N,
                   lbd, gamma, thresh, savepath, use_scm_pred=False, iterations=5, t_types='both',
                   scm_loadpath=None, scm_type=None, predict_individualized=False,
                   model_type='logreg'):
    try:
        os.mkdir(savepath)
    except OSError as err:
        print(err)
        logging.warning('Creating of directory %s failed' % savepath)
    else:
        logging.info('Creation of directory %s successful' % savepath)

    logging.info('generating problem...')

    # generate or load problem
    scm, y_name, costs = None, None, None

    if scm_loadpath is None:
        scm, y_name = generate_problem(N_nodes, p, min_in_degree, out_degree, max_uncertainty, seed=seed)
        costs = np.random.uniform(0, lbd / (N_nodes - 1), N_nodes - 1)
    else:
        scm, y_name = load_problem(scm_loadpath, type=scm_type)
        costs = np.load(scm_loadpath + 'costs.npy')


    noise = scm.sample_context(N)
    df = scm.compute()
    X = df[df.columns[df.columns != y_name]]
    y = df[y_name]

    # split into batches
    ixs = np.arange(X.shape[0])
    perc = 0.1
    ixs_test = np.random.choice(ixs, math.floor(perc * X.shape[0]), replace=False)
    ixs_train = np.delete(ixs, ixs_test)

    batches = []
    batches.append([X.iloc[ixs_train, :], y.iloc[ixs_train], noise.iloc[ixs_train, :]])
    batches.append([X.iloc[ixs_test, :], y.iloc[ixs_test], noise.iloc[ixs_test, :]])

    logging.info('Split the data into {} batches'.format(2))


    # fitting standard logistic regression on the first batch

    logging.info('Fitting model...')

    model = None
    if model_type == 'logreg':
        model = LogisticRegression()
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=5)
    else:
        raise NotImplementedError('model type {} not implemented'.format(model_type))
    model.fit(batches[0][0], batches[0][1])
    # assert model.predict_proba([[0, 0, 0, 0, 1, 0, 1, 1, 1]])[0][1] >= 0.95

    # CHECKPOINT: SAVE ALL RELEVANT DATA

    problem_setup = {'N': N, 'N_nodes': N_nodes, 'p': p, 'max_uncertainty': max_uncertainty,
                     'min_in_degree': min_in_degree,
                     'out_degree': out_degree, 'seed': seed,
                     'use_scm_pred': use_scm_pred,
                     'gamma/eta': gamma,
                     't_types': t_types,
                     'costs': list(costs)}

    logging.info('Storing all relevant data...')
    # problem setup
    with open(savepath + 'problem_setup.json', 'w') as f:
        json.dump(problem_setup, f)
    # model coefficients
    # np.save(savepath + 'model_coef.npy', np.array(model.coef_))
    # scm
    scm.save(savepath + 'scm')
    # data
    batches[0][0].to_csv(savepath + 'X_train.csv')
    batches[0][1].to_csv(savepath + 'y_train.csv')
    batches[1][0].to_csv(savepath + 'X_test.csv')
    batches[1][1].to_csv(savepath + 'y_test.csv')


    # run all types of recourse on the setting
    logging.info('Run all types of recourse...')

    r_types = ['individualized', 'subpopulation']
    t_options = ['improvement', 'acceptance']

    if t_types == 'both':
        t_types = t_options
    elif t_types in t_options:
        t_types = [t_types]

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
            result_tpl = recourse_population(scm, batches[1][0], batches[1][1], batches[1][2], y_name, costs,
                                             proportion=1.0, r_type=r_type, t_type=t_type, gamma=gamma, eta=gamma,
                                             thresh=thresh, lbd=lbd, model=model,  use_scm_pred=use_scm_pred,
                                             predict_individualized=predict_individualized)

            # save results
            logging.info('Saving results for {}_{}...'.format(t_type, r_type))
            savepath_exp = it_path
            save_recourse_result(savepath_exp, result_tpl)
            logging.info('Done.')


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
    parser.add_argument("--max_uncertainty", help="Maximum p for y node", default=0.3, type=float)
    parser.add_argument("--min_in_degree", help="minium in-degree for y_node", default=3, type=int)
    parser.add_argument("--out_degree", help="maximum out-degree for y_node", default=1, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--t_type", help="target types, either one of improvement and acceptance or both",
                        default="both", type=str)
    parser.add_argument("--scm_loadpath", help="loadpath for scm to be used", default=None, type=str)
    parser.add_argument("--scm_type", help="type of scm, either binomial or sigmoid", default='binomial', type=str)
    parser.add_argument("--predict_individualized", help="use individualized prediction if available",
                        default=False, type=bool)
    parser.add_argument("--model_type", help="model class", default='logreg', type=str)

    parser.add_argument("--logging_level", help="logging-level", default=20, type=int)

    args = parser.parse_args()

    # set logging settings
    logging.getLogger().setLevel(args.logging_level)

    # expects that we are in a directory with a subfolder called "experiments"
    # relative save paths
    config_id = random.randint(0, 1024)
    savepath_config = args.savepath + 'gamma_{}_M_{}_N_{}_id_{}/'.format(args.gamma, args.N_nodes, args.N, config_id)

    n_tries = 0
    done = False
    while n_tries < 5 and not done:
        try:
            n_tries += 1
            os.mkdir(savepath_config)
            done = True
        except Exception as err:
            logging.warning('Could not generate folder...{}'.format(savepath_config))

    run_experiment(args.N_nodes, args.p, args.max_uncertainty, args.min_in_degree, args.out_degree,
                   args.seed, args.N, args.lbd, args.gamma, args.thresh, savepath_config,
                   iterations=args.n_iterations, use_scm_pred=False, t_types=args.t_type,
                   scm_loadpath=args.scm_loadpath, scm_type=args.scm_type,
                   predict_individualized=args.predict_individualized,
                   model_type=args.model_type)
