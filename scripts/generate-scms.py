import logging
import os

import torch
from mcr.causality.scm import BinomialBinarySCM, SigmoidBinarySCM
from mcr.causality.dags import DirectedAcyclicGraph
import numpy as np

logging.getLogger().setLevel(logging.INFO)

savepath = 'scms/'

# example1

scm_dir = 'example1/'

try:
    os.mkdir(savepath + scm_dir)
except FileExistsError as err:
    logging.info('Folder already existed:' + savepath + scm_dir)
except Exception as err:
    raise err

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.05)

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
scm.set_prediction_target(y_name)

try:
    scm.save(savepath + scm_dir)
    np.save(savepath + scm_dir + 'costs.npy', costs)
except Exception as exc:
    logging.info(exc)


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



# ex2

# DEFINE DATA GENERATING MECHANISM

scm_dir = 'example2/'

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
        adjacency_matrix=np.array([[0, 1, 0, 0,],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]]),
        var_names=['vaccinated', 'covid-free', 'symptom-free', 'pcr-negative']
    ),
    p_dict={'vaccinated': sigma_high, 'covid-free': sigma_medium, 'symptom-free': sigma_medium,
            'pcr-negative': sigma_low}
)

costs = np.array([1.0, 0.1, 0.9])
y_name = 'covid-free'
scm.set_prediction_target(y_name)

try:
    scm.save(savepath + scm_dir)
    np.save(savepath + scm_dir + 'costs.npy', costs)
except Exception as exc:
    logging.info(exc)


# ex 3

# DEFINE DATA GENERATING MECHANISM

scm_dir = 'example3/'

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
        adjacency_matrix=np.array([[0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 1],
                                   [0, 0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0]]),
        var_names=['x0', 'x1', 'x2', 'y', 'x3', 'x4', 'x5']
    ),
    p_dict={'x0': sigma_medium, 'x1': sigma_medium, 'x2': sigma_medium, 'x3': sigma_medium, 'x4': sigma_medium,
            'x5': sigma_medium, 'y': sigma_medium,}
)

costs = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
y_name = 'y'
scm.set_prediction_target(y_name)

try:
    scm.save(savepath + scm_dir)
    np.save(savepath + scm_dir + 'costs.npy', costs)
except Exception as exc:
    logging.info(exc)




# ex 4

# DEFINE DATA GENERATING MECHANISM

scm_dir = 'example4/'

try:
    os.mkdir(savepath + scm_dir)
except FileExistsError as err:
    logging.info('Folder already existed:' + savepath + scm_dir)
except Exception as err:
    raise err

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.04)

scm = BinomialBinarySCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 1],
                                   [0, 0, 0, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 0]]),
        var_names=['x0', 'x1', 'x2', 'y', 'x3', 'x4', 'x5']
    ),
    p_dict={'x0': sigma_high, 'x1': sigma_high, 'x2': sigma_high, 'x3': sigma_low, 'x4': sigma_low,
            'x5': sigma_low, 'y': sigma_medium,}
)

costs = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
y_name = 'y'
scm.set_prediction_target(y_name)

try:
    scm.save(savepath + scm_dir)
    np.save(savepath + scm_dir + 'costs.npy', costs)
except Exception as exc:
    logging.info(exc)


# ex 5

# DEFINE DATA GENERATING MECHANISM

scm_dir = 'example5/'

try:
    os.mkdir(savepath + scm_dir)
except FileExistsError as err:
    logging.info('Folder already existed:' + savepath + scm_dir)
except Exception as err:
    raise err

sigma_high = torch.tensor(0.5)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.04)

scm = SigmoidBinarySCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'x3', 'x4', 'x5', 'y', 'x6', 'x7', 'x8']
    ),
    p_dict={'x1': sigma_high, 'x2': sigma_high, 'x3': sigma_high, 'x4': sigma_high, 'x5': sigma_high,
            'x6': sigma_low, 'x7': sigma_low, 'x8': sigma_low},
    sigmoid_nodes={'y'}
)

costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
y_name = 'y'
scm.set_prediction_target(y_name)

try:
    scm.save(savepath + scm_dir)
    np.save(savepath + scm_dir + 'costs.npy', costs)
except Exception as exc:
    logging.info(exc)


# ex 5

# DEFINE DATA GENERATING MECHANISM

scm_dir = 'example5_v2/'

try:
    os.mkdir(savepath + scm_dir)
except FileExistsError as err:
    logging.info('Folder already existed:' + savepath + scm_dir)
except Exception as err:
    raise err

sigma_high = torch.tensor(0.3)
sigma_medium = torch.tensor(0.09)
sigma_low = torch.tensor(0.04)

scm = SigmoidBinarySCM(
    dag=DirectedAcyclicGraph(
        adjacency_matrix=np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        var_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y', 'x7', 'x8', 'x9']
    ),
    p_dict={'x1': sigma_high, 'x2': sigma_high, 'x3': sigma_high, 'x4': sigma_high, 'x5': sigma_high,
            'x6': sigma_high, 'x7': sigma_medium, 'x8': sigma_medium, 'x9': sigma_medium},
    sigmoid_nodes={'y'},
    coeff_dict={'y': {'x1': 0.1, 'x2': 0.2, 'x3': 0.5, 'x4': 1.0, 'x5': 2.5, 'x6': 5.0}}
)

costs = np.array([0.1, 0.2, 0.5, 1.0, 2.5, 5.0, 0.1, 0.1, 0.1])
y_name = 'y'
scm.set_prediction_target(y_name)

try:
    scm.save(savepath + scm_dir)
    np.save(savepath + scm_dir + 'costs.npy', costs)
except Exception as exc:
    logging.info(exc)
