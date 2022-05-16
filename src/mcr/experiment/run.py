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
from sklearn.metrics import f1_score
import numpy as np
import math
import os

import mcr.causality.scms.examples as ex
from mcr.recourse import recourse_population, save_recourse_result
from mcr.experiment.predictors import get_tuning_rf

logging.getLogger().setLevel(20)

def run_experiment(scm_name, N, N_recourse, gamma, thresh, lbd, savepath, use_scm_pred=False, iterations=5, t_types='both',
                   seed=42, predict_individualized=False,
                   model_type='logreg', nr_refits_batch0=5, assess_robustness=False,
                   NGEN=400, POP_SIZE=1000, rounding_digits=2, tuning=False, **kwargs_model):
    try:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
    except OSError as err:
        print(err)
        logging.warning('Creating of directory %s failed' % savepath)
    else:
        logging.info('Creation of directory %s successful/directory exists already' % savepath)

    # extract SCM

    if savepath[-1] != '/':
        savepath = savepath + '/'

    if scm_name not in ex.scm_dict.keys():
        raise RuntimeError(f'SCM {scm_name} not known. Chose one of {ex.scm_dict.keys()}')
    scm = ex.scm_dict[scm_name]

    y_name, costs = scm.predict_target, np.array(scm.costs)

    # CHECKPOINT: SAVE ALL RELEVANT DATA
    logging.info('Storing all relevant data...')

    problem_setup = {'N': N, 'seed': seed,
                     'scm_name': scm_name,
                     't_types': t_types,
                     'thresh': thresh,
                     'gamma/eta': gamma,
                     'lbd': lbd,
                     'costs': list(costs),
                     'NGEN': NGEN,
                     'POP_SIZE': POP_SIZE,
                     'iterations': iterations,
                     'rounding_digits': rounding_digits,
                     'nr_refits_batch0': nr_refits_batch0,
                     'assess_robustness': assess_robustness,
                     'predict_individualized': predict_individualized,
                     'use_scm_pred': use_scm_pred,
                     'model_type': model_type,
                     }

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
    if assess_robustness:
        N_BATCHES = 3

    # find lowest experiment number that is completed
    dirs = os.listdir(savepath)
    existing_runs = 0
    for ii in range(len(dirs)):
        if f'{ii}' in dirs:
            existing_runs = ii + 1

    logging.info(f'results for up to {existing_runs} runs found')

    n_fails = 0

    # for ii in range(existing_runs, iterations):
    while existing_runs < iterations:
        logging.info('')
        logging.info('')
        logging.info('-------------')
        logging.info('ITERATION {}'.format(existing_runs))
        logging.info('-------------')


        # sample data
        noise = scm.sample_context(N)
        df = scm.compute()
        X = df[df.columns[df.columns != y_name]]
        y = df[y_name]

        # split into batches
        batch_size = math.floor(N / N_BATCHES)

        logging.info('Creating {} batches of data with {} observations'.format(N_BATCHES, batch_size))

        batches = []
        i = 0

        while i < N:
            X_i, y_i = X.iloc[i:i + batch_size, :], y.iloc[i:i + batch_size]
            U_i = noise.iloc[i:i + batch_size, :]
            batches.append((X_i, y_i, U_i))
            i += batch_size

        logging.info('Split the data into {} batches'.format(N_BATCHES))

        # fitting model on the first batch

        logging.info('Fitting model (type {})...'.format(model_type))

        model = None
        if model_type == 'logreg':
            model = LogisticRegression(penalty='none', **kwargs_model)
        elif model_type == 'rf':
            # parallelize random forest
            kwargs_model['n_jobs'] = -1
            if tuning:
                rf_random = get_tuning_rf(50, 3)

                # prepare tuning
                scm_cp = scm.copy()
                _ = scm_cp.sample_context(10**4)
                data_tuning = scm_cp.compute()
                X_tuning = data_tuning[df.columns[df.columns != y_name]]
                y_tuning = data_tuning[y_name]

                # perform tuning
                logging.info("tuning random forest parameters")
                rf_random.fit(X_tuning, y_tuning)
                rf_best_pars = rf_random.best_params_
                for par in rf_best_pars.keys():
                    if par not in kwargs_model:
                        kwargs_model[par] = rf_best_pars[par]
            if 'max_depth' not in kwargs_model:
                kwargs_model['max_depth'] = 30
            if 'n_estimators' not in kwargs_model:
                kwargs_model['n_estimators'] = 50
            if 'class_weight' not in kwargs_model:
                kwargs_model['class_weight'] = 'balanced_subsample'

            model = RandomForestClassifier(**kwargs_model)
        else:
            raise NotImplementedError('model type {} not implemented'.format(model_type))

        logging.info("fitting model with the specified parameters")
        model.fit(batches[0][0], batches[0][1])
        model_score = model.score(batches[1][0], batches[1][1])
        f1 = f1_score(batches[1][1], model.predict(batches[1][0]))
        logging.info(f"model fit with accuracy {model_score}")
        logging.info(f"f1-score {f1}")

        logging.info(f"assessing how many recourse contenders")
        pred = model.predict(batches[1][0])
        n_recourse_contendors = np.sum(pred)
        if n_recourse_contendors < N_recourse:
            logging.info(f"not enough recourse contendors ({n_recourse_contendors}). try again")
            n_fails += 1
            if n_fails >= 30:
                raise RuntimeError('Could not find enough recourse seeking individuals. ' +
                                   'Choose larger N or smaller N_recourse.')
            continue

        logging.info("enough recourse contendors. continue.")
        logging.info("create folder to starte results")
        it_path = savepath + '{}/'.format(existing_runs)
        os.mkdir(it_path)
        existing_runs += 1

        # refits for multiplicity result

        logging.info('Fitting {} models for multiplicity robustness assessment.'.format(nr_refits_batch0))
        model_refits_batch0 = []
        model_refits_batch0_scores = []
        model_refits_batch0_f1s = []
        for ii in range(nr_refits_batch0):
            model_tmp = None
            if model_type == 'logreg':
                model_tmp = LogisticRegression(penalty='none', **kwargs_model)
            elif model_type == 'rf':
                model_tmp = RandomForestClassifier(**kwargs_model)
            else:
                raise NotImplementedError('model type {} not implemented'.format(model_type))
            sample_locs = batches[0][0].sample(batches[0][0].shape[0], replace=True).index
            model_tmp.fit(batches[0][0].loc[sample_locs, :], batches[0][1].loc[sample_locs])
            model_refits_batch0.append(model_tmp)
            model_tmp_score = model_tmp.score(batches[1][0], batches[1][1])
            model_refits_batch0_scores.append(model_tmp_score)
            f1_tmp = f1_score(batches[1][1], model_tmp.predict(batches[1][0]))
            model_refits_batch0_f1s.append(f1_tmp)
            if model_type == 'logreg':
                print(model_tmp.coef_)

        # save data

        batches[0][0].to_csv(it_path + 'X_train.csv')
        batches[0][1].to_csv(it_path + 'y_train.csv')
        batches[1][0].to_csv(it_path + 'X_test.csv')
        batches[1][1].to_csv(it_path + 'y_test.csv')
        if assess_robustness:
            batches[2][0].to_csv(it_path + 'X_val.csv')
            batches[2][1].to_csv(it_path + 'y_val.csv')

        for r_type, t_type in all_combinations:
            logging.info('')
            logging.info("combination: {} {}".format(r_type, t_type))
            savepath_it_config = it_path + '{}-{}/'.format(t_type, r_type)
            os.mkdir(savepath_it_config)

            # perform recourse on batch 1
            result_tpl = recourse_population(scm, batches[1][0], batches[1][1], batches[1][2], y_name, costs,
                                             N_max=N_recourse, proportion=1.0,
                                             r_type=r_type, t_type=t_type, gamma=gamma, eta=gamma,
                                             thresh=thresh, lbd=lbd, model=model,  use_scm_pred=use_scm_pred,
                                             predict_individualized=predict_individualized,
                                             NGEN=NGEN, POP_SIZE=POP_SIZE, rounding_digits=rounding_digits)

            # save results
            logging.info('Saving results for {}_{}...'.format(t_type, r_type))
            save_recourse_result(savepath_it_config, result_tpl)
            logging.info('Done.')

            X_batch1_post_impl = result_tpl[5]
            X_batch1_post = batches[1][0].copy()
            X_batch1_post.loc[X_batch1_post_impl.index, :] = X_batch1_post_impl

            if assess_robustness:
                # create a large dataset with mixed pre- and post-recourse data
                logging.info("Create dataset mixed batch 0 pre and batch 1 post recourse")
                X_train_large = batches[0][0].copy()
                y_train_large = batches[0][1].copy()

                y_batch1_post = batches[1][1].copy()
                y_batch1_post_impl = result_tpl[6]
                y_batch1_post.loc[y_batch1_post_impl.index] = y_batch1_post_impl

                X_train_large = X_train_large.append(X_batch1_post, ignore_index=True)
                y_train_large = y_train_large.append(y_batch1_post, ignore_index=True)

                # fit a separate model on batch0_pre and batch1_post

                logging.info('Fit model on mixed dataset')
                model_post = None
                if model_type == 'logreg':
                    model_post = LogisticRegression()
                elif model_type == 'rf':
                    model_post = RandomForestClassifier(**kwargs_model)
                else:
                    raise NotImplementedError('model type {} not implemented'.format(model_type))

                model_post.fit(X_train_large, y_train_large)
                score_post = model_post.score(batches[2][0], batches[2][1])
                f1_post = f1_score(batches[2][1], model_post.predict(batches[2][0]))

                # perform recourse on batch 1
                logging.info('Perform recourse on batch 2')

                result_tpl_batch2 = recourse_population(scm, batches[2][0], batches[2][1], batches[2][2], y_name, costs,
                                                        N_max=N_recourse, proportion=1.0,
                                                        r_type=r_type, t_type=t_type,
                                                        gamma=gamma, eta=gamma,
                                                        thresh=thresh, lbd=lbd, model=model, use_scm_pred=use_scm_pred,
                                                        predict_individualized=predict_individualized,
                                                        NGEN=NGEN, POP_SIZE=POP_SIZE, rounding_digits=rounding_digits)
                X_batch2_post_impl, y_batch2_post_impl = result_tpl_batch2[5], result_tpl_batch2[6]
                recourse_recommended_ixs_batch2 = result_tpl_batch2[9]['recourse_recommended_ixs']

                # save results
                logging.info('Saving results for {}_{} batch2 ...'.format(t_type, r_type))
                savepath_batch2 = savepath_it_config + 'batch2_'
                save_recourse_result(savepath_batch2, result_tpl_batch2)
                logging.info('Done.')

                # assess acceptance for batch 2 with model_mixed
                predict_batch2 = model_post.predict(X_batch2_post_impl.loc[recourse_recommended_ixs_batch2, :])
                eta_obs_batch2 = np.mean(predict_batch2)

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
                if assess_robustness:
                    stats['eta_obs_refit'] = float(eta_obs_batch2)  # eta refit on batch0_pre and bacht1_post
                    stats['model_post_score'] = score_post
                    stats['model_post_f1'] = f1_post

                stats['eta_obs_refits_batch0_mean'] = float(np.mean(eta_obs_refits_batch0)) # mean eta of batch0-refits
                stats['model_score'] = model_score
                stats['model_f1'] = f1
                stats['model_refits_batch0_scores'] = model_refits_batch0_scores
                stats['model_refits_batch0_f1s'] = model_refits_batch0_f1s

                if model_type == 'logreg':
                    stats['model_coef'] = model.coef_.tolist()
                    stats['model_coef'].append(model.intercept_.tolist())
                    if assess_robustness:
                        stats['model_coef_refit'] = model_post.coef_.tolist()
                        stats['model_coef_refit'].append(model_post.intercept_.tolist())
                else:
                    stats['model_coef'] = float('nan')
                    if assess_robustness:
                        stats['model_coef_refit'] = float('nan')

                with open(savepath_it_config + 'stats.json', 'w') as json_file:
                    json.dump(stats, json_file)
            except Exception as exc:
                logging.info('Could not append eta_obs_batch2 to stats.json')
                logging.debug(exc)
