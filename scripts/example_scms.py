import logging
import os

import torch
from mar.causality.scm import BinomialBinarySCM
from mar.causality.dags import DirectedAcyclicGraph
import numpy as np

savepath = 'scms/'

# DEFINE DATA GENERATING MECHANISM

scm_dir = 'example1_extended/'

try:
    os.mkdir(savepath + scm_dir)
except FileExistsError as err:
    logging.info('Folder already existed:' + savepath + scm_dir)
except Exception as err:
    raise err

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.01)

scm = BinomialBinarySCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]]),
        var_names=['vaccinated', 'contacts', 'mask', 'covid-free', 'symptom-free']
    ),
    p_dict={'vaccinated': sigma_high, 'contacts': sigma_high, 'mask': sigma_high,
            'symptom-free': sigma_low, 'covid-free': sigma_medium}
)

costs = np.array([1.0, 1.0, 1.0, 0.1])
y_name = 'covid-free'
scm.set_prediction_target(y_name)

try:
    scm.save(savepath + scm_dir)
    np.save(savepath + scm_dir + 'costs.npy', costs)
except Exception as exc:
    logging.info(exc)