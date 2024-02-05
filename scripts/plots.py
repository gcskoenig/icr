import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rc
import argparse

sns.set_context('paper')
rc('font', **{'family': 'serif', 'serif': ['Bookman']})

from icr.experiment.compile import compile_experiments

scm_names = {'3var-causal': '3var-c-collected', '3var-noncausal': '3var-nc-collected',
             '7var-covid': '7var-covid-collected',
             '5var-skill': '5var-skill-collected'
             }
levels = ['gam0.75_', 'gam0.85_', 'gam0.9_', 'gam0.95_']

def has_levels(string):
    has = False
    for level in levels:
        has = has or level in string
    return has

dfss = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")

    parser.add_argument("--savepath", help="path to savepath", type=str, default='../experiments/remote-experiments/neurips/')
    args = parser.parse_args()

    basepath = args.savepath

    for scm_name in scm_names.keys():
        loadpath = basepath + scm_names[scm_name] + '/'
        compile_experiments(loadpath, scm_name)
        dirs = os.listdir(loadpath)
        dirs = [d for d in dirs if d[0] == 'N']
        dirs = [d for d in dirs if has_levels(d)]
        dfs = []
        for dir in dirs:
            df = pd.read_csv(loadpath + dir + '/result_raw.csv')
            dfs.append(df)

        dfs = pd.concat(dfs, ignore_index=True)
        dfs['scm'] = scm_name
        dfss.append(dfs)

    dfss = pd.concat(dfss, ignore_index=True)
    dfss['confidence'] = dfss['gamma'].copy()
    dfss.loc[dfss['t_type'] == 'improvement', 't_type'] = 'ICR'
    dfss.loc[dfss['t_type'] == 'acceptance', 't_type'] = 'CR'
    dfss.loc[dfss['t_type'] == 'counterfactual', 't_type'] = 'CE'

    # create dataset of interest with columns rtype ttype value value-type confidence

    value_types = ['gamma_obs', 'eta_obs_individualized', 'eta_obs', 'eta_obs_refits_batch0_mean', 'intv-cost']#, 'perc_recomm_found']
    value_names = {value_types[0]: r'$\gamma^{obs}$',
                   value_types[1]: r'$\eta^{obs, model}$',
                   value_types[2]: r'$\eta^{obs, model}$',
                   value_types[3]: r'$\eta^{obs, refit}$',
                   value_types[4]: r'\text{cost}'#,
                   #value_types[4]: '%found_recomm'
                   }
    info_columns = ['scm', 't_type', 'r_type']

    #value_types = [value_types[2]]

    def compile_df_plt(dfss, value_types, value_names):
        dfs_plt = []
        for value in value_types:
            df_plt = dfss[info_columns].copy()
            df_plt['method'] = df_plt['t_type'].copy()
            df_plt['type'] = df_plt['r_type'].copy()
            values = dfss[value].to_numpy()
            df_plt['confidence'] = dfss['gamma']
            df_plt['value'] = values
            df_plt['metric'] = value_names[value] + ' ' + df_plt['method'].copy()
            if value == 'eta_obs_individualized':
                ixs_indMCR = np.logical_and(df_plt['method'] == 'ICR', df_plt['type'] == 'individualized')
                df_plt = df_plt.loc[ixs_indMCR, :].copy()
                df_plt['type'] = 'ind. prediction'
            dfs_plt.append(df_plt)

        df_plt = pd.concat(dfs_plt, ignore_index=True)
        df_plt.loc[df_plt['t_type'] == 'CE', 'r_type'] = 'other'
        df_plt.loc[df_plt['t_type'] == 'CE', 'confidence'] = 0.95
        df_cp = df_plt.loc[df_plt['t_type'] == 'CE', :].copy()
        df_cp.loc[df_plt['t_type'] == 'CE', 'confidence'] = 0.75
        df_plt = pd.concat([df_plt, df_cp], ignore_index=True)

        ixs_indiv = np.logical_and(df_plt['t_type'] == 'ICR', df_plt['r_type'] == 'individualized')
        ixs_indiv = np.logical_and(ixs_indiv, df_plt['metric'] == 'gamma_obs ICR')
        df_plt_ind = df_plt.loc[ixs_indiv, :].copy()

        df_plt = df_plt.sort_values(['metric', 't_type'])
        return df_plt


    df_plt = compile_df_plt(dfss, value_types, value_names)
    ixs_indiv = df_plt['metric'] == '$\eta^{obs, ind}$ ICR'

    df_plt = compile_df_plt(dfss, ['gamma_obs'], value_names)

    sns.set_style('whitegrid')
    ticks = [0.0, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0]
    xticks = [0.75, 0.85, 0.9, 0.95]
    ylim = [0.0, 1.005]

    metric_sets = {'improvement': ['gamma_obs'],
                   'acceptance': ['eta_obs', 'eta_obs_individualized'],
                   'acceptance_refit': ['eta_obs_refits_batch0_mean']}
    ncol = 6
    tickright = False

    for name in metric_sets.keys():
        df_plt = compile_df_plt(dfss, metric_sets[name], value_names)

        with sns.plotting_context('talk'):

            legend = False
            if name == 'improvement':
                legend = False
                # legend = True


            g2 = sns.relplot(data=df_plt, x="confidence", y="value", col='metric', style='type',
                             hue='scm', kind='line', markers=True, dashes=False, err_kws={'alpha': 0.1},
                             col_wrap=min(len(df_plt['metric'].unique()), ncol),
                             palette=sns.color_palette("tab10", len(df_plt['scm'].unique())),
                             height=5, aspect=0.6,
                             legend=legend, alpha=0.7,
                             ci='sd',
                             facet_kws={'sharex': False, 'sharey': False})
            g2.set_titles('{col_name}')
            g2.set_axis_labels(r'confidence', r'')
            for ii in range(len(g2.axes.flat)):
                ax = list(g2.axes.flat)[ii]
                if ii % ncol == ncol - 1:
                    ax.yaxis.tick_right()
                title = ax.title.get_text()
                if not 'cost' in title:
                    ax.set_yscale('function', functions=[lambda x: x ** 2, lambda z: np.power(z, 1 / 2)])
                    ax.set_ylim(*ylim)
                    ax.set_yticks(ticks)
                    if ii % ncol == 0 or (ii % ncol == ncol - 1 and tickright):
                        ax.set_yticklabels(ticks)
                    else:
                        ax.set_yticklabels([])
                else:
                    ax.set_ylim(0, dfss['intv-cost'].max())
                    if ii % 3 < 2:
                        ax.set_yticklabels([])
                    #ax.set_yticks([0, dfss['intv-cost'].max()])
                    #ax.set_yticklabels([0, 'max'])
                if not 'CE' in title:
                    ax.set_xscale('function', functions=[lambda x: x**2, lambda z: np.power(z, 1/2)])
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticks)
                else:
                    ax.set_xscale('function', functions=[lambda x: x ** 2, lambda z: np.power(z, 1 / 2)])
                    ax.set_xticks([0.75, 0.95])
                    ax.set_xticklabels([])
                new_title = ''
                if 'ICR' in title:
                    new_title += 'ICR'
                    ax.set_xlabel('$\overline{\gamma}$')
                elif 'CR' in title:
                    new_title += 'CR'
                    ax.set_xlabel('$\overline{\eta}$')
                else:
                    new_title += 'CE'
                    ax.set_xlabel('')
                if 'gamma' in title:
                    if 'refit' in title:
                        new_title += ': $\gamma^{obs, refit}$'
                    else:
                        new_title += ': $\gamma^{obs}$'
                elif 'eta' in title:
                    if 'refit' in title:
                        new_title += ': $\eta^{obs, refit}$'
                    else:
                        new_title += ': $\eta^{obs}$'
                elif 'cost' in title:
                    new_title += ': cost'

                ax.set_title(new_title)

            sns.despine(left=True)
            #sns.move_legend(ax, "center right")
            plt.subplots_adjust(wspace=0.2, hspace=0.4)
            plt.savefig(basepath + 'summary_' + name + '.pdf', bbox_inches='tight')
            plt.show()


    metrics = ['gamma_obs', 'eta_obs', 'eta_obs_refits_batch0_mean', 'eta_obs_individualized',
               'intv-cost', 'perc_recomm_found']
    settings = ['scm', 't_type', 'r_type', 'gamma']
    dfss_sub = dfss[settings + metrics]
    result_table = dfss_sub.groupby(settings).mean()
    result_table = result_table[metrics]
    result_table = result_table.sort_values(settings)
    result_table_std = dfss_sub.groupby(settings).std()
    result_table_std = result_table_std[metrics]
    result_table_std = result_table_std.sort_values(settings)

    table_all = result_table.merge(result_table_std, left_index=True, right_index=True,
                                   suffixes=('_mean', '_std'))
    column_order = ['gamma_obs_mean', 'gamma_obs_std', 'eta_obs_mean', 'eta_obs_std',
                    'eta_obs_individualized_mean', 'eta_obs_individualized_std',
                    'eta_obs_refits_batch0_mean_mean',
                    'eta_obs_refits_batch0_mean_std',
                    'intv-cost_mean',
                    'intv-cost_std'
                    ]
    # table_all = table_all.reindex(sorted(table_all.columns), axis=1)
    table_all = table_all[column_order].copy()
    table_all = table_all.round(2)
    table_all.to_csv(basepath + 'table_all.csv')

    dfs = table_all.copy()
    dfs = dfs.reset_index()
    order_new = ['name', 'gamma'] + (column_order)
    dfs['name'] = dfs['t_type'] + dfs['r_type']

    def df_to_tex_table(df, index=False, header=False):
        df = df.to_string(index=index, header=header).split('\n')
        df = ['  &'.join(ele.split()) for ele in df]
        df = [element.replace('NaN', '-') for element in df]
        df = '\\\\ \n'.join(df)
        return df

    scms = list(dfs['scm'].unique())
    scm = '3var-noncausal'
    for scm in scms:
        df_sub = dfs.loc[dfs['scm'] == scm, order_new].copy()
        df = df_to_tex_table(df_sub)
        with open(basepath + 'tex-result-' + scm + '.txt', 'w') as f:
            f.write(df)

    res_ce = dfs.loc[dfs['t_type'] == 'CE', :].groupby(['scm']).mean().reset_index().round(2)
    s = df_to_tex_table(res_ce)
    with open(basepath + 'tex-result-ce' + '.txt', 'w') as f:
        f.write(s)