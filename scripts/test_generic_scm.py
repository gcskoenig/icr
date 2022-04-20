from mcr.causality.scm import GenericSCM
from mcr.causality.dags import DirectedAcyclicGraph
import os
import numpy as np
import logging
import torch

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.05)
sigma_verylow = torch.tensor(0.001)

scm = GenericSCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 0, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]]),
        var_names=['vaccinated', 'covid-free', 'symptom-free', 'entrance', 'assigned-building']
    )
)

costs = np.array([0.5, 0.1, 0.1, 0.1])
y_name = 'covid-free'
scm.set_prediction_target(y_name)