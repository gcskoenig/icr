from mcr.causality.scm import GenericSCM
from mcr.causality.dags import DirectedAcyclicGraph
import os
import numpy as np
import logging
import torch

logging.getLogger().setLevel(logging.DEBUG)

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.05)
sigma_verylow = torch.tensor(0.001)

def sigmoidal_fnc(df_par, df_noise):
    assert df_noise.shape[1] == 1
    mean_noise = df_noise.to_numpy() # .mean(axis=1)
    mean_pars = df_par.to_numpy().mean(axis=1)
    result = np.zeros(mean_noise.shape[0])
    if mean_pars.shape[0] > 0:
        result += mean_pars
    result = torch.sigmoid(torch.tensor(result))
    output = torch.tensor(mean_noise).flatten() <= result.flatten()
    return output


scm = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 0, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]]),
        var_names=['vaccinated', 'covid-free', 'symptom-free', 'entrance', 'assigned-building']
    ),
    fnc_dict={'covid-free': sigmoidal_fnc}
)

costs = np.array([0.5, 0.1, 0.1, 0.1])
y_name = 'covid-free'
scm.set_prediction_target(y_name)

scm.sample_context(1000)
data = scm.compute()

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data)
plt.show()