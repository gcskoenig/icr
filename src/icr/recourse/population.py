import logging
import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from icr.causality.utils import indvd_to_intrv
from icr.recourse.recourse import recourse
from icr.recourse.utils import compute_h_post_individualized


def recourse_population(scm, X, y, U, y_name, costs, proportion=1.0, N_max=None, nsamples=10**3, r_type='individualized',
                        t_type='acceptance', gamma=0.7, eta=0.7, thresh=0.5, lbd=1.0, subpopulation_size=500,
                        model=None, use_scm_pred=False, predict_individualized=False, NGEN=400, POP_SIZE=1000,
                        rounding_digits=2):
    assert not (model is None and not use_scm_pred)

    if N_max is None:
        N_max = len(y)

    # initializing prediction setup
    predict_log_proba = None
    if use_scm_pred:
        scm.set_prediction_target(y_name)
        predict_log_proba = scm.predict_log_prob
    else:
        predict_log_proba = model.predict_log_proba

    logging.debug('Determining rejected individuals and individuals determined to implement recourse...')
    predictions = np.exp(predict_log_proba(X)[:, 1]).flatten() >= thresh
    ixs_rejected = np.arange(len(predictions))[predictions == 0]
    ixs_recourse = np.random.choice(ixs_rejected, size=math.floor(proportion * len(ixs_rejected)), replace=False)

    if len(ixs_recourse) > N_max:
        ixs_recourse = ixs_recourse[:N_max]

    logging.debug('Detected {} rejected and {} recourse seeking individuals...'.format(len(ixs_rejected),
                                                                                       len(ixs_recourse)))

    intv_features = X.columns
    if t_type == 'improvement':
        causes_dag = list(scm.dag.get_ancestors_node(y_name))
        causes = [nd for nd in intv_features if nd in causes_dag]  # to make sure the ordering is as desired
        ixs_causes = np.array([np.arange(len(intv_features))[cause == intv_features][0] for cause in causes])
        costs = costs[ixs_causes]
        intv_features = causes

    X_new = X.copy()
    y_new = None
    if y is not None:
        y_new = y.copy()
    interventions = []
    goal_costs = []
    intv_costs = []

    logging.debug('Iterating through {} individuals to suggest recourse...'.format(len(ixs_recourse)))
    for ix in tqdm(ixs_recourse):
        obs = X.iloc[ix, :]

        scm_ = None

        # for individualized recourse abduction is performed at this step
        if r_type == 'subpopulation' or t_type == 'counterfactual':
            scm_ = scm.copy()
        elif r_type == 'individualized':
            scm_ = scm.abduct(obs, n_samples=nsamples)
        else:
            raise NotImplementedError('r_type must be in {}'.format(['individualized', 'subpopulation']))

        # compute optimal action
        cntxt = scm_.sample_context(size=nsamples)
        winner, goal_cost, intv_cost = recourse(scm_, intv_features, obs, costs, r_type, t_type,
                                                predict_log_proba=predict_log_proba, y_name=y_name,
                                                gamma=gamma, eta=eta, thresh=thresh, lbd=lbd,
                                                subpopulation_size=subpopulation_size, NGEN=NGEN, POP_SIZE=POP_SIZE,
                                                rounding_digits=rounding_digits, multi_objective=False,
                                                return_stats=False, X=X)

        intervention = indvd_to_intrv(scm, intv_features, winner, obs)

        interventions.append(winner)
        goal_costs.append(goal_cost)
        intv_costs.append(intv_cost)

        # compute the actual outcome for this observation
        scm_true = scm.copy()
        u_tmp = U.iloc[ix, :].to_dict()
        scm_true.set_noise_values(u_tmp)
        sample = scm_true.compute(do=intervention)
        X_new.iloc[ix, :] = sample[X.columns].to_numpy()
        y_new.iloc[ix] = sample[y_name]

    logging.debug('Collecting results...')
    interventions = np.array(interventions)
    interventions = pd.DataFrame(interventions, columns=intv_features)
    interventions['ix'] = X.index[ixs_recourse]
    interventions.set_index('ix', inplace=True)

    costss = np.array([goal_costs, intv_costs]).T
    costss = pd.DataFrame(costss, columns=['goal_cost', 'intv_cost'])
    costss.index = interventions.index.copy()

    ixs_rp = interventions[np.abs(interventions.abs().sum(axis=1)) > 0].index  # indexes where recourse was performed

    X_pre = X.iloc[ixs_recourse, :]
    y_pre = y.iloc[ixs_recourse]
    X_post = X_new.iloc[ixs_recourse, :]
    y_post = y_new.iloc[ixs_recourse]

    logging.debug('Collecting pre- and post-recourse model predictions...')
    y_hat_pre = pd.DataFrame(predictions[ixs_recourse], columns=['y_hat'])
    h_post = predict_log_proba(X_post[X.columns])[:, 1].flatten()
    h_post = pd.DataFrame(h_post, columns=['h_post'])
    h_post['h_post_individualized'] = np.nan
    h_post = np.exp(h_post)
    h_post.index = y_post.index.copy()

    if r_type == 'individualized' and t_type == 'improvement' and predict_individualized:
        logging.debug('Computing individualized post-recourse predictions...')
        h_post_indiv = compute_h_post_individualized(scm, X_pre, X_post, interventions, intv_features, y_name, y=1)
        h_post['h_post_individualized'] = h_post_indiv['h_post_individualized']

    for df in [X_post, y_post, interventions, X_pre, y_pre, y_hat_pre, h_post]:
        df.index = X.index[ixs_recourse]

    # compute model performance on pre- and post-recourse data
    predict_pre = model.predict(X_pre)
    predict_post = model.predict(X_post)
    accuracy_pre = accuracy_score(y_pre, predict_pre)
    accuracy_post = accuracy_score(y_post, predict_post)


    logging.debug('Computing stats...')
    stats = {}
    stats['accuracy_pre'] = accuracy_pre
    stats['accuracy_post'] = accuracy_post
    stats['recourse_seeking_ixs'] = list(interventions.index.copy())
    stats['recourse_recommended_ixs'] = list(ixs_rp.copy())
    stats['perc_recomm_found'] = float(ixs_rp.shape[0] / X_post.shape[0])
    stats['gamma'] = float(gamma)
    stats['eta'] = float(eta)
    stats['gamma_obs'] = float(y_post[ixs_rp].mean())
    stats['gamma_obs_pre'] = float(y_pre[ixs_rp].mean())
    eta_obs = (h_post.loc[ixs_rp, :] >= thresh).mean()
    stats['eta_obs'] = float(eta_obs['h_post'])
    stats['costs'] = list(costs)  # costs for the interventions (list with len(X.columns) indexes)
    stats['lbd'] = float(lbd)
    stats['thresh'] = float(thresh)
    stats['r_type'] = str(r_type)
    stats['t_type'] = str(t_type)

    if not h_post['h_post_individualized'].hasnans:
        stats['eta_obs_individualized'] = eta_obs['h_post_individualized']
    else:
        stats['eta_obs_individualized'] = np.nan

    logging.debug('Done.')
    return U, X_pre, y_pre, y_hat_pre, interventions, X_post, y_post, h_post, costss, stats