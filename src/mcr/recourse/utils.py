import json
import logging

import numpy as np
import pandas as pd
import torch

from mcr.causality.utils import indvd_to_intrv


def compute_h_post_individualized(scm, X_pre, X_post, invs, features, y_name, y=1):
    """
    Computes the individualized post-recourse predictions (probabilities)
    """
    log_probs = np.zeros(invs.shape[0])
    for ix in range(invs.shape[0]):
        intv_dict = indvd_to_intrv(scm, features, invs.iloc[ix, :], X_pre.iloc[0, :])
        log_probs[ix] = torch.exp(scm.predict_log_prob_individualized_obs(X_pre.iloc[ix, :], X_post.iloc[ix, :],
                                                                          intv_dict, y_name, y=y))
    h_post_individualized = pd.DataFrame(log_probs, columns=['h_post_individualized'])
    h_post_individualized.index = X_pre.index.copy()
    return h_post_individualized


def save_recourse_result(savepath_exp, result_tupl):
    U, X_pre, y_pre, y_hat_pre, invs, X_post, y_post, h_post, costss, stats = result_tupl
    U.to_csv(savepath_exp + 'U.csv')
    X_pre.to_csv(savepath_exp + 'X_pre.csv')
    y_pre.to_csv(savepath_exp + 'y_pre.csv')
    y_hat_pre.to_csv(savepath_exp + 'y_hat_pre.csv')
    invs.to_csv(savepath_exp + 'invs.csv')
    X_post.to_csv(savepath_exp + 'X_post.csv')
    y_post.to_csv(savepath_exp + 'y_post.csv')
    h_post.to_csv(savepath_exp + 'h_post.csv')
    costss.to_csv(savepath_exp + 'costss.csv')

    try:
        with open(savepath_exp + 'stats.json', 'w') as f:
            json.dump(stats, f)
    except Exception as exc:
        logging.warning('stats.json could not be saved.')
        logging.info('Exception: {}'.format(exc))
