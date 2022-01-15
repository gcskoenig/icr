import pandas as pd
import numpy as np
import os

import logging
import argparse

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Aggregate experiment results")

    parser.add_argument("savepath",
                        help="savepath for the experiment folder. either relative to working directory or absolute.",
                        type=str)

    args = parser.parse_args()

    base_base_path = args.savepath

    dirs = [ name for name in os.listdir(base_base_path) if os.path.isdir(os.path.join(base_base_path, name)) ]

    for dir in dirs:
        base_path = base_base_path + dir +'/'

        r_types = ['individualized', 'subpopulation']
        t_types = ['improvement', 'acceptance']

        cols = ['eta_obs_model', 'eta_obs_refit', 'r_type', 't_type']
        df = pd.DataFrame([], columns=cols)

        for r_type in r_types:
            for t_type in t_types:
                path = base_path
                path = path + '{}_{}/'.format(t_type, r_type)

                try:
                    pred1 = np.loadtxt(path + 'predict0.csv')
                    pred2 = np.loadtxt(path + 'predict1.csv')

                    res = {'eta_obs_model' : np.mean(pred1), 'eta_obs_refit': np.mean(pred2),
                           'r_type': r_type, 't_type': t_type}

                    df = df.append(res, ignore_index=True)
                except Exception as err:
                    logging.warning('Could not load file {} or {}'.format(path + 'predict0.csv', 'predict1.csv'))

        try:
            result_dir = base_path
            df.to_csv(result_dir + 'aggregated_result.csv')
            logging.info('SUCCESS in folder {}'.format(dir))
        except Exception as err:
            logging.info('Not successful in directory {}'.format(dir))
            logging.info(err)