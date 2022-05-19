import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rc

sns.set_context('talk')
#sns.set(rc={'text.usetex': True})
rc('font',**{'family':'serif','serif':['Bookman']})

from mcr.experiment.compile import compile_experiments
from mcr.causality.scms.examples import scm_dict

scm_names = {'3var-causal': '3var-c-4', '3var-noncausal': '3var-nc-4',
             '5var-skill': '5var-skill-3', '7var-covid': '7var-covid-2'}
levels = ['gam0.75_', 'gam0.85_', 'gam0.9_', 'gam0.95_']

def has_levels(string):
    has = False
    for level in levels:
        has = has or level in string
    return has

dfss = []

basepath = '../experiments/remote-experiments/neurips/'

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
dfss.loc[dfss['t_type'] == 'improvement', 't_type'] = 'MCR'
dfss.loc[dfss['t_type'] == 'acceptance', 't_type'] = 'CR'
dfss.loc[dfss['t_type'] == 'counterfactual', 't_type'] = 'CE'

# create dataset of interest with columns rtype ttype value value-type confidence

value_types = ['gamma_obs', 'eta_obs']#, 'eta_obs_refits_batch0_mean', 'intv-cost']
value_names = {value_types[0]: r'$\gamma^{obs, model}$', value_types[1]: r'$\eta^{obs, model}$',
               value_types[2]: r'$\eta^{obs,refit}$', value_types[3]: r'\text{cost}'}
info_columns = ['scm', 't_type', 'r_type']

#value_types = [value_types[2]]

dfs_plt = []
for value in value_types:
    df_plt = dfss[info_columns].copy()
    df_plt['method'] = df_plt['t_type'].copy()
    df_plt['type'] = df_plt['r_type'].copy()
    values = dfss[value].to_numpy()
    df_plt['confidence'] = dfss['gamma']
    df_plt['value'] = values
    df_plt['metric'] = value_names[value] + ' ' + df_plt['method'].copy()
    dfs_plt.append(df_plt)

df_plt = pd.concat(dfs_plt, ignore_index=True)

df_plt.loc[df_plt['t_type'] == 'CE', 'r_type'] = 'other'
df_plt.loc[df_plt['t_type'] == 'CE', 'confidence'] = 0.95
df_cp = df_plt.loc[df_plt['t_type'] == 'CE', :].copy()
df_cp.loc[df_plt['t_type'] == 'CE', 'confidence'] = 0.75
df_plt = pd.concat([df_plt, df_cp], ignore_index=True)


#df_plt['hue'] = df_plt['r_type'] + '-' + df_plt['t_type']


df_plt_cf = df_plt.loc[df_plt['t_type'] == 'CE', :].copy()
df_plt_cf.loc[df_plt_cf['metric'] == 'gamma_obs CE', 'metric'] = '$\gamma^{obs}$'
df_plt_cf.loc[df_plt_cf['metric'] == 'eta_obs CE', 'metric'] = '$\eta^{obs}$'

df_plt_wocf = df_plt.loc[df_plt['t_type'] != 'CE', :].copy()


df_plt = df_plt.sort_values(['metric', 't_type'])


sns.set_style('whitegrid')
ticks = [0.0, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0]
xticks = [0.75, 0.85, 0.9, 0.95]
ylim = [-0.1, 1.1]


with sns.plotting_context('talk'):


    g2 = sns.relplot(data=df_plt, x="confidence", y="value", col='metric', style='type',
                     hue='scm', kind='line', markers=True, dashes=False, err_kws={'alpha': 0.1},
                     col_wrap=6, palette=sns.color_palette("tab10", 4), height=4, aspect=0.6,
                     legend=True, alpha=0.7,
                     facet_kws={'sharex': False, 'sharey': False})
    g2.set_titles('{col_name}')
    g2.set_axis_labels(r'confidence', r'')
    for ii in range(len(g2.axes.flat)):
        ax = list(g2.axes.flat)[ii]
        if ii % 6 == 5:
            ax.yaxis.tick_right()
        title = ax.title.get_text()
        if not 'cost' in title:
            ax.set_yscale('function', functions=[lambda x: x ** 2, lambda z: np.power(z, 1 / 2)])
            ax.set_ylim(*ylim)
            ax.set_yticks(ticks)
            if ii % 6 == 0 or ii % 6 == 5:
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
        if 'MCR' in title:
            new_title += 'MCR'
            ax.set_xlabel('$\gamma$')
        elif 'CR' in title:
            new_title += 'CR'
            ax.set_xlabel('$\eta$')
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
    plt.subplots_adjust(wspace=0.17, hspace=0.4)
    plt.savefig(basepath + 'summary.pdf', bbox_inches='tight')
    plt.show()