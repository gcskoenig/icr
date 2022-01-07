import pandas as pd

from mar.causality.scm import LinearGaussianNoiseSCM, BinomialBinarySCM
from mar.causality.dags import DirectedAcyclicGraph
import torch
import logging
from sklearn.linear_model import LogisticRegression

from mar.recourse import recourse_population, compute_h_post_individualized

import numpy as np

logging.getLogger().setLevel(logging.INFO)

sigma_medium = 0.5
sigma_low = 0.1

scm = LinearGaussianNoiseSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1],
                                   [0, 0, 0],
                                   [0, 0, 0]]),
        var_names=['x1', 'x2', 'y']
    ),
    coeff_dict={'x2': {'x1': 1.0}, 'y': {'x1': 1.0}},
    noise_std_dict={'x1': sigma_medium, 'x2': sigma_medium, 'y': sigma_low}
)


# DEFINE DATA GENERATING MECHANISM

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.3)
sigma_low = torch.tensor(0.2)

scm = BinomialBinarySCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 0]]),
        var_names=['vaccinated', 'covid-free', 'symptom-free']
    ),
    p_dict={'vaccinated': sigma_high,
            'symptom-free': sigma_low, 'covid-free': sigma_medium}
)

costs = np.array([0.5, 0.1])
y_name = 'covid-free'

# GENERATE DATA AND PERFORM RECOURSE

N = 30

noise = scm.sample_context(N)
df = scm.compute()

X = df[df.columns[df.columns != y_name]]
y = df[y_name]

model = LogisticRegression()
model.fit(X, y)

X_pre, y_pre, y_hat_pre, invs, X_post, y_post, h_post, stats = recourse_population(scm, model, X, y, noise, y_name,
                                                                                   costs, proportion=1.0,
                                                                                   r_type='individualized',
                                                                                   t_type='improvement',
                                                                                   gamma=0.9, lbd=10.0,
                                                                                   thresh=0.5)
stats
