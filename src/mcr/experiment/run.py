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
import numpy as np
import math
import random
import os
import argparse

import mcr.causality.examples as ex
from mcr.recourse import recourse_population, save_recourse_result

from mcr.experiment.compile import compile_experiments

logging.getLogger().setLevel(20)

def run_experiment(scm_name, N, gamma, thresh, lbd, savepath, use_scm_pred=False, iterations=5, t_types='both',
                   seed=42, predict_individualized=False,
                   model_type='logreg', nr_refits_batch0=5, **kwargs_model):
    try:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
    except OSError as err:
        print(err)
        logging.warning('Creating of directory %s failed' % savepath)
    else:
        logging.info('Creation of directory %s successful/directory exists already' % savepath)

    # extract SCM

    if scm_name not in ex.scm_dict.keys():
        raise RuntimeError(f'SCM {scm_name} not known. Chose one of {ex.scm_dict.keys()}')
    scm = ex.scm_dict[scm_name]

    y_name, costs = scm.predict_target, np.array(scm.costs)

    # CHECKPOINT: SAVE ALL RELEVANT DATA

    problem_setup = {'N': N, 'seed': seed,
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
    # scm.save(savepath + 'scm')

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


    N_BATCHES = 2

    for ii in range(iterations):
        logging.info('')
        logging.info('')
        logging.info('-------------')
        logging.info('ITERATION {}'.format(ii))
        logging.info('-------------')

        it_path = savepath + '{}/'.format(ii)
        os.mkdir(it_path)

        # sample data
        noise = scm.sample_context(N)
        df = scm.compute()
        X = df[df.columns[df.columns != y_name]]
        y = df[y_name]

        # split into batches
        batch_size = math.floor(N / N_BATCHES)

        logging.info('Creating three batches of data with {} observations'.format(batch_size))

        batches = []
        i = 0

        while i < N:
            X_i, y_i = X.iloc[i:i + batch_size, :], y.iloc[i:i + batch_size]
            U_i = noise.iloc[i:i + batch_size, :]
            batches.append((X_i, y_i, U_i))
            i += batch_size

        logging.info('Split the data into {} batches'.format(3))

        # fitting model on the first batch

        logging.info('Fitting model...')

        model = None
        if model_type == 'logreg':
            model = LogisticRegression(penalty='none', **kwargs_model)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=1, max_depth=2, **kwargs_model)
        else:
            raise NotImplementedError('model type {} not implemented'.format(model_type))
        model.fit(batches[0][0], batches[0][1])

        # refits for multiplicity result

        logging.info('Fitting {} models for multiplicity robustness assessment.'.format(nr_refits_batch0))
        model_refits_batch0 = []
        for ii in range(nr_refits_batch0):
            model_tmp = None
            if model_type == 'logreg':
                model_tmp = LogisticRegression(penalty='none', **kwargs_model)
            elif model_type == 'rf':
                model_tmp = RandomForestClassifier(n_estimators=1, max_depth=2, **kwargs_model)
            else:
                raise NotImplementedError('model type {} not implemented'.format(model_type))
            sample_locs = batches[0][0].sample(batches[0][0].shape[0], replace=True).index
            model_tmp.fit(batches[0][0].loc[sample_locs, :], batches[0][1].loc[sample_locs])
            model_refits_batch0.append(model_tmp)
            if model_type == 'logreg':
                print(model_tmp.coef_)

        # save data

        batches[0][0].to_csv(it_path + 'X_train.csv')
        batches[0][1].to_csv(it_path + 'y_train.csv')
        batches[1][0].to_csv(it_path + 'X_test.csv')
        batches[1][1].to_csv(it_path + 'y_test.csv')
        # batches[2][0].to_csv(it_path + 'X_val.csv')
        # batches[2][1].to_csv(it_path + 'y_val.csv')

        for r_type, t_type in all_combinations:
            logging.info('')
            logging.info("combination: {} {}".format(r_type, t_type))
            savepath_it_config = it_path + '{}-{}/'.format(t_type, r_type)
            os.mkdir(savepath_it_config)

            # perform recourse on batch 1
            result_tpl = recourse_population(scm, batches[1][0], batches[1][1], batches[1][2], y_name, costs,
                                             proportion=1.0, r_type=r_type, t_type=t_type, gamma=gamma, eta=gamma,
                                             thresh=thresh, lbd=lbd, model=model,  use_scm_pred=use_scm_pred,
                                             predict_individualized=predict_individualized)

            # save results
            logging.info('Saving results for {}_{}...'.format(t_type, r_type))
            save_recourse_result(savepath_it_config, result_tpl)
            logging.info('Done.')

            # # create a large dataset with mixed pre- and post-recourse data
            # logging.info("Create dataset mixed batch 0 pre and batch 1 post recourse")
            # X_train_large = batches[0][0].copy()
            # y_train_large = batches[0][1].copy()

            X_batch1_post = batches[1][0].copy()
            # y_batch1_post = batches[1][1].copy()
            # X_batch1_post_impl = result_tpl[5]
            # y_batch1_post_impl = result_tpl[6]
            # X_batch1_post.loc[X_batch1_post_impl.index, :] = X_batch1_post_impl
            # y_batch1_post.loc[y_batch1_post_impl.index] = y_batch1_post_impl
            #
            # X_train_large = X_train_large.append(X_batch1_post, ignore_index=True)
            # y_train_large = y_train_large.append(y_batch1_post, ignore_index=True)
            #
            # # fit a separate model on batch0_pre and batch1_post
            #
            # logging.info('Fit model on mixed dataset')
            # model_post = None
            # if model_type == 'logreg':
            #     model_post = LogisticRegression()
            # elif model_type == 'rf':
            #     model_post = RandomForestClassifier(n_estimators=5)
            # else:
            #     raise NotImplementedError('model type {} not implemented'.format(model_type))
            #
            # model_post.fit(X_train_large, y_train_large)
            #
            # # perform recourse on batch 1
            # logging.info('Perform recourse on batch 2')
            #
            # result_tpl_batch2 = recourse_population(scm, batches[2][0], batches[2][1], batches[2][2], y_name, costs,
            #                                         proportion=1.0, r_type=r_type, t_type=t_type, gamma=gamma, eta=gamma,
            #                                         thresh=thresh, lbd=lbd, model=model, use_scm_pred=use_scm_pred,
            #                                         predict_individualized=predict_individualized)
            # X_batch2_post_impl, y_batch2_post_impl = result_tpl_batch2[5], result_tpl_batch2[6]
            # recourse_recommended_ixs = result_tpl_batch2[9]['recourse_recommended_ixs']
            #
            # # save results
            # logging.info('Saving results for {}_{} batch2 ...'.format(t_type, r_type))
            # savepath_batch2 = savepath_it_config + 'batch2_'
            # save_recourse_result(savepath_batch2, result_tpl_batch2)
            # logging.info('Done.')
            #
            # # assess acceptance for batch 2 with model_mixed
            # predict_batch2 = model_post.predict(X_batch2_post_impl.loc[recourse_recommended_ixs, :])
            # eta_obs_batch2 = np.mean(predict_batch2)

            # access acceptance for batch 1 with multiplicity models (without distribution shift)
            eta_obs_refits_batch0 = []
            recourse_recommended_ixs_batch1 = result_tpl[9]['recourse_recommended_ixs']
            for ii in range(nr_refits_batch0):
                predict_batch1 = model_refits_batch0[ii].predict(X_batch1_post.loc[recourse_recommended_ixs_batch1, :])
                eta_obs_refit_batch0 = np.mean(predict_batch1)
                eta_obs_refits_batch0.append(eta_obs_refit_batch0)

            # save additional stats in the stats.json
            logging.info('Saving additional stats.')
            try:
                with open(savepath_it_config + 'stats.json') as json_file:
                    stats = json.load(json_file)

                # add further information to the statistics
                # stats['eta_obs_refit'] = float(eta_obs_batch2)  # eta refit on batch0_pre and bacht1_post
                stats['eta_obs_refits_batch0_mean'] = float(np.mean(eta_obs_refits_batch0)) # mean eta of batch0-refits

                if model_type == 'logreg':
                    stats['model_coef'] = model.coef_.tolist()
                    stats['model_coef'].append(model.intercept_.tolist())
                    # stats['model_coef_refit'] = model_post.coef_.tolist()
                    # stats['model_coef_refit'].append(model_post.intercept_.tolist())
                else:
                    stats['model_coef'] = float('nan')
                    # stats['model_coef_refit'] = float('nan')

                with open(savepath_it_config + 'stats.json', 'w') as json_file:
                    json.dump(stats, json_file)
            except Exception as exc:
                logging.info('Could not append eta_obs_batch2 to stats.json')
                logging.debug(exc)


if __name__ == '__main__':
    # parsing command line arguments
    parser = argparse.ArgumentParser("Create recourse experiments. " +
                                     "For every configuration a separate folder is created. " +
                                     "Within every folder a folder for every interation is created." +
                                     "The savepath specifies the folder in which these folders shall be placed.")

    parser.add_argument("scm_name", help=f"one of {ex.scm_dict.keys()}", type=str)
    parser.add_argument("savepath",
                        help="savepath for the experiment folder. either relative to working directory or absolute.",
                        type=str)
    parser.add_argument("gamma", help="gammas for recourse", type=float)
    parser.add_argument("N", help="Number of observations", type=int)
    parser.add_argument("n_iterations", help="number of runs per configuration", type=int)

    parser.add_argument("--thresh", help="threshs for prediction and recourse", type=float, default=0.5)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--t_type", help="target types, either one of improvement and acceptance or both",
                        default="both", type=str)
    parser.add_argument("--scm_type", help="type of scm, either binomial or sigmoid", default='binomial', type=str)
    parser.add_argument("--predict_individualized", help="use individualized prediction if available",
                        default=True, type=bool)
    parser.add_argument("--model_type", help="model class", default='logreg', type=str)

    parser.add_argument("--logging_level", help="logging-level", default=20, type=int)
    parser.add_argument("--ignore_np_errs", help="whether to ignore all numpy warnings and errors",
                        default=True, type=bool)

    args = parser.parse_args()

    # set logging settings
    logging.getLogger().setLevel(args.logging_level)

    if args.ignore_np_errs:
        np.seterr(all="ignore")

    savepath_config = None


    n_tries = 0
    done = False
    while n_tries < 5 and not done:
        try:
            config_id = random.randint(0, 1024)
            savepath_config = args.savepath + 'gamma_{}_M_{}_N_{}_id_{}/'.format(args.gamma, args.N_nodes, args.N,
                                                                                 config_id)
            n_tries += 1
            os.mkdir(savepath_config)
            done = True
        except Exception as err:
            logging.warning('Could not generate folder...{}'.format(savepath_config))

    run_experiment(args.scm_name, args.N, args.lbd, args.gamma, args.thresh, savepath_config,
                   seed=args.seed,
                   iterations=args.n_iterations, use_scm_pred=False, t_types=args.t_type,
                   predict_individualized=args.predict_individualized,
                   model_type=args.model_type)

    compile_experiments(args.savepath)
