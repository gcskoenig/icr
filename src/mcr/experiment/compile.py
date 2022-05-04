import pandas as pd
import json
import os
import mcr.causality.examples as ex

import logging
import argparse

from mcr.causality.scms import BinomialBinarySCM

logging.getLogger().setLevel(logging.INFO)

def get_dirs(savepath):
    dirs = [name for name in os.listdir(savepath) if os.path.isdir(os.path.join(savepath, name))]
    dirs = [os.path.join(savepath, dir, '') for dir in dirs]
    return dirs

def compile_experiments(savepath, dirs=None, assess_robustness=False, scm_name=None):
    base_base_path = savepath

    if dirs is None:
        dirs = get_dirs(savepath)

    cols_cost = ['r_type', 't_type', 'iteration', 'intv-cost']
    cols = ['perc_recomm_found', 'gamma', 'eta', 'gamma_obs', 'gamma_obs_pre', 'eta_obs', 'eta_obs_individualized',
            'costs', 'lbd', 'thresh', 'r_type', 't_type', 'iteration', 'model_coef',
            'eta_obs_refits_batch0_mean']
    output_cols = ['eta', 'gamma', 'perc_recomm_found', 'eta_obs', 'eta_obs_individualized',
                   'eta_obs_refits_batch0_mean', 'gamma_obs', 'intv-cost']
    if assess_robustness:
        cols = cols + ['eta_obs_refit', 'model_coef_refit']
        output_cols = output_cols + ['eta_obs_refit']


    df_resultss = pd.DataFrame([])
    df_invs_resultss = pd.DataFrame([])

    for dir in dirs:
        base_path = dir
        path = base_path

        # load scm
        try:
            scm = ex.scm_dict[scm_name]
            causes = scm.dag.get_ancestors_node(scm.predict_target)
            non_causes = set(scm.dag.var_names) - causes - {scm.predict_target}
        except FileNotFoundError as err:
            logging.warning(err)
            break


        r_types = ['individualized', 'subpopulation']
        t_types = ['improvement', 'acceptance']

        # RESULT DATAFRAMES
        df = pd.DataFrame([], columns=cols)
        df_cost = pd.DataFrame([], columns=cols_cost)
        df_invs = pd.DataFrame([])
        df_coefs = pd.DataFrame([])
        df_coefs_refits = pd.DataFrame([])

        # loop over iterations/experiments to fill the result dataframes
        it_dirs = [int(name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        n_iterations = len(it_dirs)
        for it in it_dirs:
            path_it = path + '{}/'.format(it)
            for r_type in r_types:
                for t_type in t_types:
                    path_it_config = path_it + '{}-{}/'.format(t_type, r_type)

                    # try to read the stats.json
                    try:
                        f = None
                        if 'batch2_stats.json' in os.listdir(path_it_config):
                            f = open(path_it_config + 'batch2_stats.json')
                        elif 'stats.json' in os.listdir(path_it_config):
                            f = open(path_it_config + 'stats.json')
                        else:
                            raise FileNotFoundError('Neither stats.json nor batch2_stats.json found.')
                        stats = json.load(f)
                        f.close()

                        stats['iteration'] = it
                        stats_series = pd.Series(stats)
                        df = df.append(stats_series[cols], ignore_index=True)

                        if not type(stats_series['model_coef']) is list:
                            coefs = pd.Series(stats_series['model_coef'][0] + stats_series['model_coef'][1])

                            coefs['t_type'] = t_type
                            coefs['r_type'] = r_type
                            coefs['it'] = int(it)

                            df_coefs = df_coefs.append(coefs, ignore_index=True)

                            if assess_robustness:
                                coefs_refit = pd.Series(stats_series['model_coef_refit'][0] +
                                                        stats_series['model_coef_refit'][1])

                                coefs_refit['t_type'] = t_type
                                coefs_refit['r_type'] = r_type
                                coefs_refit['it'] = int(it)

                                df_coefs_refits = df_coefs_refits.append(coefs_refit, ignore_index=True)

                    except Exception as err:
                        logging.warning('Could not load file[s] {}'.format(path_it_config + '[batch2_]stats.json'))

                    # try to read and extract information from costss.csv and invs.csv
                    # to build cost overview
                    try:
                        cost_tmp = pd.read_csv(path_it_config + 'costss.csv', index_col=0)
                        invs_tmp = pd.read_csv(path_it_config + 'invs.csv', index_col=0)

                        ixs_recourse_recommended = invs_tmp.index[(invs_tmp.mean(axis=1) > 0)]
                        cost = cost_tmp.loc[ixs_recourse_recommended, 'intv_cost'].mean()
                        df_cost = df_cost.append(
                            {'r_type': r_type, 't_type': t_type, 'iteration': it, 'intv-cost': cost},
                            ignore_index=True)
                    except Exception as err:
                        logging.warning('Could not load file {}'.format(path_it_config + 'costss.csv'))

                    # try to read invs.csv to get itnervention type overview
                    try:
                        invs = pd.read_csv(path_it_config + 'invs.csv', index_col=0)
                        ixs_recourse = invs.index[invs.sum(axis=1) > 0]
                        invs = invs.loc[ixs_recourse, :]
                        invs = invs.mean(axis=0).to_dict()
                        invs['t_type'] = t_type
                        invs['r_type'] = r_type
                        invs['iteration'] = it
                        df_invs = df_invs.append(invs, ignore_index=True)
                    except Exception as err:
                        logging.warning('Could not load file {}'.format(path_it_config + 'invs.csv'))


        try:
            # join the dataframes
            df = df.join(df_cost, lsuffix='', rsuffix='_cost')
            df.drop(list(df.filter(regex='_cost$')), axis=1, inplace=True)

            # get mean and standard deviation
            groupby_cols = ['r_type', 't_type']

            # main table
            gb_obj = df.groupby(groupby_cols)
            mean_table = gb_obj.mean()[output_cols]
            # if n_iterations > 1:
            std_table = gb_obj.agg(lambda x: x.std())[output_cols]
            result_table = mean_table.join(std_table, lsuffix='_mean', rsuffix='_std')
            # else:
            #     result_table = mean_table
            #     result_table.columns = result_table.columns + '_mean'

            # invs table
            df_invs = df_invs.loc[:, df_invs.columns != 'iteration']
            df_invs.loc[:, 'causes'] = df_invs.loc[:, causes].sum(axis=1)
            df_invs.loc[:, 'non-causes'] = df_invs.loc[:, non_causes].sum(axis=1)
            gb_obj = df_invs.groupby(groupby_cols)
            invs_mean = gb_obj.mean()
            # if n_iterations > 1:
            invs_std = gb_obj.agg(lambda x: x.std())
            invs_res = invs_mean.join(invs_std, lsuffix='_mean', rsuffix='_std')
            # else:
            #     invs_res = invs_mean
            #     invs_res.columns = invs_res.columns + '_mean'


            result_dir = base_path
            result_table.to_csv(result_dir + 'aggregated_result.csv')
            invs_res.to_csv(result_dir + 'aggregated_invs.csv')
            logging.info('SUCCESS in folder {}'.format(dir))

            invs_res['gamma'] = result_table['gamma_mean'][0]
            df_resultss = pd.concat([df_resultss, result_table])
            df_invs_resultss = pd.concat([df_invs_resultss, invs_res])
        except Exception as err:
            logging.info('Not successful in directory {}'.format(dir))
            logging.info(err)

    df_resultss = df_resultss.sort_values(['t_type', 'r_type', 'gamma_mean'])
    df_resultss.to_csv(base_base_path + 'resultss.csv')

    df_invs_resultss = df_invs_resultss.sort_values(['t_type', 'r_type', 'gamma'])
    df_invs_resultss.to_csv(base_base_path + 'invs_resultss.csv')

    if len(df_coefs.columns) > 0:
        df_coefs.set_index(['t_type', 'r_type', 'it'], inplace=True)

        df_coefs_mean = df_coefs.groupby(['t_type', 'r_type']).mean()
        df_coefs.to_csv(base_base_path + 'model_coefs.csv')
        df_coefs_mean.to_csv(base_base_path + 'model_coefs_mean.csv')

        if assess_robustness:
            df_coefs_refits.set_index(['t_type', 'r_type', 'it'], inplace=True)
            df_coefs_refits_mean = df_coefs_refits.groupby(['t_type', 'r_type']).mean()
            df_coefs_refits.to_csv(base_base_path + 'model_coefs_refits.csv')
            df_coefs_refits_mean.to_csv(base_base_path + 'model_coefs_refits_mean.csv')


# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser("Aggregate experiment results")
#
#     parser.add_argument("savepath",
#                         help="savepath for the experiment folder. either relative to working directory or absolute.",
#                         type=str)
#
#     args = parser.parse_args()
#     compile_experiments(args.savepath)
