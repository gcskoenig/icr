import pandas as pd
import json
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

        n_iterations = 5
        r_types = ['individualized', 'subpopulation']
        t_types = ['improvement', 'acceptance']

        cols = ['perc_recomm_found', 'gamma', 'eta', 'gamma_obs', 'gamma_obs_pre', 'eta_obs', 'costs', 'lbd', 'thresh',
                'r_type', 't_type', 'iteration']
        df = pd.DataFrame([], columns=cols)

        for r_type in r_types:
            for t_type in t_types:
                for it in range(n_iterations):
                    path = base_path
                    path = path + '{}-{}/'.format(t_type, r_type)
                    path = path + '{}/'.format(it)

                    try:
                        f = None
                        if '_stats.json' in os.listdir(path):
                            f = open(path + '_stats.json')
                        elif 'stats.json' in os.listdir(path):
                            f = open(path + 'stats.json')
                        else:
                            raise FileNotFoundError('Neither stats.json nor _stats.json found.')
                        stats = json.load(f)
                        f.close()

                        stats['iteration'] = it
                        stats_series = pd.Series(stats)
                        df = df.append(stats_series[cols], ignore_index=True)
                    except Exception as err:
                        logging.warning('Could not load file {}'.format(path + '_stats.json'))

        try:
            # get mean and standard deviation
            groupby_cols = ['r_type', 't_type']
            output_cols = ['eta', 'gamma', 'perc_recomm_found', 'eta_obs', 'gamma_obs']
            gb_obj = df.groupby(groupby_cols)

            mean_table = gb_obj.mean()[output_cols]
            std_table = gb_obj.std()[output_cols]

            result_table = mean_table.join(std_table, lsuffix='_mean', rsuffix='_std')

            result_dir = base_path
            result_table.to_csv(result_dir + 'aggregated_result.csv')
            logging.info('SUCCESS in folder {}'.format(dir))
        except Exception as err:
            logging.info('Not successful in directory {}'.format(dir))
            logging.info(err)