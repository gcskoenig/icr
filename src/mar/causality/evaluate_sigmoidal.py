import torch
from mar.causality.scm import SigmoidBinarySCM
from mar.causality.dags import DirectedAcyclicGraph
import numpy as np

savepath = 'scms/'

# ex 5

# DEFINE DATA GENERATING MECHANISM

scm_dir = 'example5/'

# try:
#     os.mkdir(savepath + scm_dir)
# except FileExistsError as err:
#     logging.info('Folder already existed:' + savepath + scm_dir)
# except Exception as err:
#     raise err

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

scm.sample_context(100)
# scm.compute_node('y')
data = scm.compute()

obs = data.iloc[0, :]
obs = obs.loc[obs.index != 'y']

scm_abd = scm.abduct(obs)
smpl = scm_abd.sample_context(100)
smpl_data = scm_abd.compute()



from mar.backend.dist import TransformedUniform

dist = TransformedUniform(0.7, 0.9)

sample = dist.rsample(sample_shape=(10000,))

import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(sample)
plt.show()

import numpy as np
import torch
xs = np.arange(0, 1, 0.01)
xs = torch.tensor(xs)

probs = dist.log_prob(xs)

plt.scatter(xs, torch.exp(probs))
plt.show()