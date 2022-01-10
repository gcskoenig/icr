import pandas as pd
import json

import logging

logging.getLogger().setLevel(logging.INFO)


experiment_path = '/Users/gcskoenig/university/phd/2021/research/paper_2021_robust_recourse/experiments/'
experimental_setting = 'random_all_types/'
config_folder = 'gamma-something/'


n_iterations = 5
r_types = ['individualized', 'subpopulation']
t_types = ['improvement', 'acceptance']

cols = ['perc_recomm_found', 'gamma', 'eta', 'gamma_obs', 'gamma_obs_pre', 'eta_obs', 'costs', 'lbd', 'thresh', 'r_type', 't_types']
df = pd.DataFrame([], columns=cols)

for r_type in r_types:
    for t_type in t_types:
        for it in range(n_iterations):
            path = experiment_path + experimental_setting + config_folder
            path = path + '{}-{}/'.format(t_type, r_type)
            path = path + '{}/'.format(it)

            try:
                f = open(path + 'stats.json')
                stats = json.load(f)
                f.close()

                stats['iteration'] = it
                df = df.append(stats[cols], ignore_index=True)
            except Exception as err:
                logging.warning('Could not load file {}'.format(path + 'stats.json'))


# aggregate per type of recourse

# get mean and standard deviation

# probably makes more sense to look into that as soon as the rest has been done

# yuhuu