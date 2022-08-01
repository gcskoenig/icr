from mcr.causality.scms.examples import scm_dict
from mcr.recourse import recourse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from mcr.causality.utils import indvd_to_intrv
import pandas as pd
import argparse
from tqdm import tqdm

scms = ['3var-causal', '3var-noncausal', '5var-skill', '7var-covid']
model_dict = {'3var-causal': 'logreg', '3var-noncausal': 'logreg',
              '5var-skill': 'rf',
              '7var-covid': 'rf'}
scms = ['7var-covid']

use_scm_pred = False
lbd = 5000

N = 10000
test_size = round(0.3 * N)

gammas = [0.5, 0.75, 0.85, 0.90, 0.95]

results = {}

scm_name = scms[0]

N_INDIVID = 5
thresh = 0.5
SUBPOPULATION_SIZE = 10**3
MAX_EVAL = 20

if __name__ == "__main__":
    # savepath = '../experiments/eval-acceptance-subp/'

    parser = argparse.ArgumentParser("")

    parser.add_argument("savepath", help="path to savepath", type=str)
    parser.add_argument("--N", help="number of runs", type=int, default=N_INDIVID)

    args = parser.parse_args()

    savepath = args.savepath
    N_INDIVID = args.N

    # iterate over the different scms
    for scm_name in scms:
        print(scm_name)

        df_res = []

        scm = scm_dict[scm_name]

        context = scm.sample_context(N)
        data = scm.compute()

        cols = data.columns[data.columns != scm.predict_target]
        X = data.loc[:, cols]
        y = data.loc[:, scm.predict_target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        if model_dict[scm_name] == 'rf':
            model = RandomForestClassifier()
        elif model_dict[scm_name] == 'logreg':
            model = LogisticRegression()
        else:
            raise RuntimeError('model class not available')

        model.fit(X_train, y_train)

        ixs_rejected = np.exp(model.predict_log_proba(X_test)[:, 1]) <= thresh
        ixs_rejected = np.arange(len(X_test))[ixs_rejected]

        # for each scm compute recourse recommendation(s) for N_INDIV individuals
        for gamma in gammas:
            print(f'\t{scm_name}, gamma: {gamma}')
            for ii in tqdm(range(N_INDIVID)):
                # print(f'individual: {ii}')

                ix = ixs_rejected[ii]

                obs = X_test.iloc[ix, :]

                result_tupl = recourse(scm, cols, obs, scm.costs, 'subpopulation', 'improvement',
                                       predict_log_proba=model.predict_log_proba, gamma=gamma,
                                       multi_objective=False, y_name=scm.predict_target,
                                       thresh=thresh, X=X_test,
                                       POP_SIZE=400, NGEN=800, lbd=lbd)

                winner, goal_cost, intv_cost = result_tupl
                # print(winner)

                indivduals = np.array([winner])
                values = np.array([[goal_cost]])

                # we already have the meaningfulness estimates
                # we additionally need the subpopulation-based acceptance rate
                # and the acceptance bound

                df_res_ind = []

                # iterate over individuals to get the quantities
                for jj in range(min(MAX_EVAL, len(indivduals))):

                    ind = indivduals[jj, :]
                    meaningfulness = values[jj, 0]

                    # sample the subgroub
                    scm_cp = scm.copy()
                    acs = scm_cp.dag.get_ancestors_node(scm_cp.predict_target)
                    intv_dict = indvd_to_intrv(scm_cp, cols, ind, obs)
                    intv_dict_causes = {k: intv_dict[k] for k in acs & intv_dict.keys()}
                    scm_cp = scm_cp.fix_nondescendants(intv_dict_causes, obs)
                    scm_cp.sample_context(SUBPOPULATION_SIZE)
                    data = scm_cp.compute(do=intv_dict)

                    # compute acceptance rate
                    pred_proba = np.exp(model.predict_log_proba(data.loc[:, cols]))[:, 1]
                    mean_proba = np.mean(pred_proba)
                    acceptance_rate = np.mean(pred_proba >= thresh)

                    if use_scm_pred:
                        # use scm_prediction
                        preds_scm = []
                        for kk in range(len(data)):
                            pred_proba_scm = scm.predict_log_prob_obs(data.iloc[kk, :], scm.predict_target)
                            preds_scm.append(np.exp(pred_proba_scm))
                        preds_scm = np.array(preds_scm)

                        # compute acceptance rate
                        mean_proba_scm = np.mean(preds_scm)
                        acceptance_rate_acm = np.mean(preds_scm > thresh)


                    # compile to observation
                    res_model = {'gamma': meaningfulness, 'eta_emp': acceptance_rate,
                                 'mean_proba': mean_proba, 'type': 'model', 'gamma_spec': gamma}
                    res_model = dict(res_model, **intv_dict_causes)

                    res = pd.Series(res_model)
                    df_res_ind.append(res)


                    if use_scm_pred:
                        res_scm = {'gamma': meaningfulness, 'eta_emp': acceptance_rate_acm,
                                   'mean_proba': mean_proba_scm, 'type': scm_name, 'gamma_spec': gamma}
                        res_scm = dict(res_scm, **intv_dict_causes)
                        res_scm = pd.Series(res_scm)
                        df_res_ind.append(res_scm)

                df_res_ind = pd.concat(df_res_ind, axis=1, ignore_index=True).T
                df_res_ind['id'] = jj
                df_res.append(df_res_ind)

        df_res = pd.concat(df_res, ignore_index=True)

        results[scm_name] = df_res

    results[scm_name] = pd.concat([results[scm_name], df_res], ignore_index=True)

    import seaborn as sns
    import matplotlib.pyplot as plt

    def get_bound(gamma, thresh):
        return (gamma - thresh) / (1 - thresh)

    def convert_data(df, except_cols=None):
        if except_cols is None:
            cols = df.columns
        else:
            ls = list(df.columns)
            for col in except_cols:
                if col in ls:
                    ls.pop(ls.index(col))
            cols = ls
        for col in cols:
            df[col] = np.array(df[col], dtype=np.float64)
        return df

    df_all = []
    for scm in scms:
        df = convert_data(results[scm], except_cols=['type', 'eta_emp_type', 'scm'])
        df_2 = df.copy()
        df_2['eta_emp'] = get_bound(df['gamma_spec'], thresh)
        df['eta_emp_type'] = 'emp'
        df_2['eta_emp_type'] = 'bound'
        df = pd.concat([df, df_2])
        df['scm'] = scm
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)

    results.to_csv(savepath + 'results.csv')

    sns.lineplot(data=df_all.reset_index(), x='gamma_spec', y='eta_emp',
                 hue='eta_emp_type', style='scm')
    plt.savefig(savepath + 'lineplot.pdf')
    plt.show()

    df_all.to_csv(savepath + 'df_all.csv')

