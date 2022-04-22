from mcr.causality.scm import GenericSCM
from mcr.causality.dags import DirectedAcyclicGraph
import os
import numpy as np
import logging
import torch

logging.getLogger().setLevel(logging.INFO)

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.05)
sigma_verylow = torch.tensor(0.001)

def sigmoidal_fnc(x_pa, u_j):
    result = u_j
    if x_pa.shape[0] > 0:
        mean_pars = torch.mean(x_pa, axis=1)
        result = u_j + mean_pars
    result = torch.sigmoid(result)
    return result.flatten()

scm = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 0, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]]),
        var_names=['vaccinated', 'covid-free', 'symptom-free', 'entrance', 'assigned-building']
    )#,
    #fnc_dict={'covid-free': sigmoidal_fnc}
)

costs = np.array([0.5, 0.1, 0.1, 0.1])
y_name = 'covid-free'
scm.set_prediction_target(y_name)

context = scm.sample_context(1000)
data = scm.compute()

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.pairplot(data)
# plt.show()

scm_abd = scm.abduct_node('vaccinated', data.iloc[0, :], infer_type='mcmc')
# # scm_abd_svi = scm.abduct_node('covid-free', data.iloc[0, :], infer_type='svi')
print(context.iloc[0, :])
scm_abd.mean