import pandas as pd
import json
import os

import logging
import argparse

from mcr.causality.scm import BinomialBinarySCM

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Aggregate experiment results")

    parser.add_argument("savepath",
                        help="savepath for the experiment folder. either relative to working directory or absolute.",
                        type=str)

    args = parser.parse_args()

    cols = ['perc_recomm_found', 'gamma', 'eta', 'gamma_obs', 'gamma_obs_pre', 'eta_obs', 'eta_obs_individualized',
            'costs', 'lbd', 'thresh', 'r_type', 't_type', 'iteration']
    cols_cost = ['r_type', 't_type', 'iteration', 'intv-cost']
    output_cols = ['eta', 'gamma', 'perc_recomm_found', 'eta_obs', 'eta_obs_individualized',
                   'gamma_obs', 'intv-cost']

    df_resultss = pd.DataFrame([])
    df_invs_resultss = pd.DataFrame([])

    base_base_path = args.savepath

    dirs = [ name for name in os.listdir(base_base_path) if os.path.isdir(os.path.join(base_base_path, name)) ]

    for dir in dirs:
        base_path = base_base_path + dir +'/'

        # load scm
        scm = BinomialBinarySCM.load(base_path + 'scm')
        causes = scm.dag.get_ancestors_node(scm.predict_target)
        non_causes = set(scm.dag.var_names) - causes - {scm.predict_target}

        n_iterations = 5
        r_types = ['individualized', 'subpopulation']
        t_types = ['improvement', 'acceptance']

        df = pd.DataFrame([], columns=cols)
        df_cost = pd.DataFrame([], columns=cols_cost)
        df_invs = pd.DataFrame([])

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
                        cost_tmp = pd.read_csv(path + 'costss.csv', index_col=0)
                        invs_tmp = pd.read_csv(path + 'invs.csv', index_col=0)
                        ixs_recourse_recommended = invs_tmp.index[(invs_tmp.mean(axis=1) > 0)]
                        cost = cost_tmp.loc[ixs_recourse_recommended, 'intv_cost'].mean()
                        df_cost = df_cost.append({'r_type': r_type, 't_type': t_type, 'iteration': it, 'intv-cost': cost},
                                       ignore_index=True)
                    except Exception as err:
                        logging.warning('Could not load file {}'.format(path + 'costss.csv'))

                    try:
                        invs = pd.read_csv(path + 'invs.csv', index_col=0)
                        ixs_recourse = invs.index[invs.sum(axis=1) > 0]
                        invs = invs.loc[ixs_recourse, :]
                        invs = invs.mean(axis=0).to_dict()
                        invs['t_type'] = t_type
                        invs['r_type'] = r_type
                        invs['iteration'] = it
                        df_invs = df_invs.append(invs, ignore_index=True)
                    except Exception as err:
                        logging.warning('Could not load file {}'.format(path + 'invs.csv'))

        try:
            # join the dataframes
            df = df.join(df_cost, lsuffix='', rsuffix='_cost')

            # get mean and standard deviation
            groupby_cols = ['r_type', 't_type']

            # main table

            gb_obj = df.groupby(groupby_cols)
            mean_table = gb_obj.mean()[output_cols]
            std_table = gb_obj.std()[output_cols]
            result_table = mean_table.join(std_table, lsuffix='_mean', rsuffix='_std')

            # invs table
            df_invs = df_invs.loc[:, df_invs.columns != 'iteration']
            df_invs.loc[:, 'causes'] = df_invs.loc[:, causes].sum(axis=1)
            df_invs.loc[:, 'non-causes'] = df_invs.loc[:, non_causes].sum(axis=1)
            gb_obj = df_invs.groupby(groupby_cols)
            invs_mean = gb_obj.mean()
            invs_std = gb_obj.std()
            invs_res = invs_mean.join(invs_std, lsuffix='_mean', rsuffix='_std')

            result_dir = base_path
            result_table.to_csv(result_dir + 'aggregated_result.csv')
            invs_res.to_csv(result_dir + 'aggregated_invs.csv')
            logging.info('SUCCESS in folder {}'.format(dir))

            invs_res['gamma'] = result_table['gamma_mean'][0]
            df_resultss = df_resultss.append(result_table)
            df_invs_resultss = df_invs_resultss.append(invs_res)
        except Exception as err:
            logging.info('Not successful in directory {}'.format(dir))
            logging.info(err)


    df_resultss = df_resultss.sort_values(['t_type', 'r_type', 'gamma_mean'])
    df_resultss.to_csv(base_base_path + 'resultss.csv')

    df_invs_resultss = df_invs_resultss.sort_values(['t_type', 'r_type', 'gamma'])
    df_invs_resultss.to_csv(base_base_path + 'invs_resultss.csv')
