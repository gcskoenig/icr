import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import seaborn as sns

import os
import re
import argparse

levels = ['gam0.75_', 'gam0.85_', 'gam0.9_', 'gam0.95_']

def has_levels(string):
    has = False
    for level in levels:
        has = has or level in string
    return has


scm_names = {'3var-causal': '3var-c-collected', '3var-noncausal': '3var-nc-collected',
             '7var-covid': '7var-covid-collected',
             '5var-skill': '5var-skill-collected'
             }

basepath = '../results/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")

    parser.add_argument("--savepath", help="path to savepath", type=str, default='../experiments/remote-experiments/neurips/')
    parser.add_argument("--ex_type", help="individualized or subpopulation", type=str, default="individualized")
    args = parser.parse_args()
    basepath = args.savepath
    rtype = args.ex_type

    scm_name = list(scm_names.keys())[0]
    files = []
    for scm_name in scm_names.keys():
        loadpath = basepath + scm_names[scm_name] + '/'
        dirs = os.listdir(loadpath)
        dirs = [d for d in dirs if d[0] == 'N']
        dirs = [d for d in dirs if has_levels(d)]
        dfs = []
        dir = dirs[0]
        for dir in dirs:
            subdir = loadpath + dir + '/'
            _files = glob.glob(subdir + f'*/improvement-{rtype}/h_post.csv')
            files = files + _files

    file = files[0]
    mean_scores = []
    scores = []
    for file in files:
        m = re.search('_gam(.+?)_', file)
        gamma_level = float(m.group(1))

        m = re.search(basepath + '(.+?)/N', file)
        scm = m.group(1)

        h_post = pd.read_csv(file, index_col=0)

        folder = os.path.dirname(file)
        gammas = pd.read_csv(folder + '/costss.csv', index_col=0).goal_cost

        h_post['gamma'] = gammas
        h_post['gamma_target'] = gamma_level
        h_post['scm'] = scm

        scores.append(h_post)

        # tpl = (scm, gamma_level,
        #        h_post['h_post'].mean(), h_post['h_post_individualized'].mean(),
        #        np.mean(h_post['h_post'] >= 0.5), np.mean(h_post['h_post_individualized'] >= 0.5))
        #
        # mean_scores.append(tpl)

        # h_post_ = pd.DataFrame()
        # col = h_post.columns[0]
        # for col in ['h_post', 'h_post_individualized']:
        #     df = pd.DataFrame()
        #     df['value'] = h_post[col]
        #     df['predictor'] = col
        #     h_post_ = pd.concat([h_post_, df], ignore_index=True)

        # plt.figure()
        # g = sns.displot(data=h_post_, x='value', row='predictor')
        #
        # def specs(x, **kwargs):
        #     plt.axvline(x.mean(), c='k', ls='-', lw=2.5)
        #     plt.axvline(gamma_level, c='orange', ls='--', lw=2.5)
        #
        # g.map(specs, 'value')
        #
        # plt.savefig(file + '_plot.pdf')
        # plt.show()

    # arr = np.array(mean_scores)
    # cols = ['scm', 'gamma', 'mean_pre', 'mean_post', 'eta_pre', 'eta_post']
    #
    # df = pd.DataFrame(arr, columns=cols)
    #
    # for col in cols[1:]:
    #     df.loc[:, col] = pd.to_numeric(df.loc[:, col])

    scoress = pd.concat(scores, ignore_index=True)
    scoress['acc_post'] = scoress['h_post'] >= 0.5
    scoress['acc_post_individualized'] = scoress['h_post_individualized'] >= 0.5
    # scoress_rounded = scoress.round(decimals=2)
    scoress_rounded = scoress.copy()
    steps = 4  # e.g. 4 if you want 0.25 steps or 2 if 0.5 steps
    scoress_rounded['gamma'] = (scoress['gamma'] * steps).round(decimals=1) / steps

    ## filter out those where more than 50 values recorded

    def remove_comb(df, scm, gamma):
        df = df.loc[np.logical_or(df['scm'] != scm, df['gamma'] != gamma), :].copy()
        return df

    def identify_seldom(df, cols, thresh):
        count_series = df.groupby(cols).size()
        cnt_df = count_series.to_frame(name='size').reset_index()
        res = cnt_df.loc[cnt_df['size'] < thresh, cols]
        res = [list(res.iloc[i]) for i in range(res.shape[0])]
        return res

    def remove_seldom(df, cols, thresh):
        seldom = identify_seldom(df, cols, thresh)
        for sel in seldom:
            print('remove sel')
            print(df.shape)
            df = remove_comb(df, sel[0], sel[1])
            print(df.shape)
        return df


    scoress_subset = remove_seldom(scoress_rounded, ['scm', 'gamma'], 20)

    g = sns.lineplot(data=scoress_subset, x='gamma', y='h_post', hue='scm',
                     n_boot=1000)
    g.axes.axline((0.75, 0.75), (0.95, 0.95), linestyle='--', color='gray')
    plt.show()

    g = sns.lineplot(data=scoress_subset, x='gamma', y='acc_post', hue='scm',
                     n_boot=1000)
    g.axes.axline((0.75, 0.75), (0.95, 0.95), linestyle='--', color='gray')
    plt.show()