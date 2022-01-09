import pandas as pd
import json



config_path = '/Users/gcskoenig/university/phd/2021/research/paper_2021_robust_recourse/experiments/random_all_types/config_gamma0.5_thresh0.5_Nnodes10_p0.8_860/'
n_iterations = 1

cols = ['perc_recomm_found', 'gamma', 'eta', 'gamma_obs', 'gamma_obs_pre', 'eta_obs', 'costs', 'lbd', 'thresh', 'r_type', 't_types']
df = pd.DataFrame([], columns=cols)

# compile results
for ii in range(n_iterations):
    it_path = config_path + '{}/'.format(ii)
    f = open(it_path + '_stats.json')
    stats = json.load(f)
    f.close()

    df = df.append(stats[cols], ignore_index=True)

df

