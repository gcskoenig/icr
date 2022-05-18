import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

sns.set_context('talk')
#sns.set(rc={'text.usetex': True})

from mcr.experiment.compile import compile_experiments
from mcr.causality.scms.examples import scm_dict

scm_names = {'3var-causal': '3var-c-3', '3var-noncausal': '3var-nc-3', '7var-covid': '7var-covid-2'}
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


# create dataset of interest with columns rtype ttype value value-type confidence

value_types = ['gamma_obs', 'eta_obs']
value_names = {value_types[0]: r'gamma_obs', value_types[1]: r'eta_obs'}
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
    df_plt['metric'] = df_plt['method'].copy() + ': ' + value_names[value]
    dfs_plt.append(df_plt)

df_plt = pd.concat(dfs_plt, ignore_index=True)

#df_plt['hue'] = df_plt['r_type'] + '-' + df_plt['t_type']


sns.set_style('whitegrid')

g = sns.relplot(data=df_plt, x="confidence", y="value", col='metric', style='type',
                hue='scm', kind='line', markers=True, dashes=False, err_kws={'alpha': 0.1},
                col_wrap=4, palette=sns.color_palette("tab10", 3), aspect=0.5
                )
g.set_titles('{col_name}')
g.set_axis_labels(r'gamma or eta', r'improvement/acceptance rate')
ticks = [0.0, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0]
xticks = [0.75, 0.85, 0.9, 0.95]
for ax in g.axes.flat:
    ax.set_ylim(0, 1)
    ax.set_yscale('function', functions=[lambda x: x**2, lambda z: np.power(z, 1/2)])
    ax.set_xscale('function', functions=[lambda x: x**2, lambda z: np.power(z, 1/2)])
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
sns.despine(left=True)
plt.subplots_adjust(wspace=0.2)
plt.savefig(basepath + 'summary.pdf')
plt.show()