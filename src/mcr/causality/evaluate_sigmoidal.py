import torch
from mcr.causality.scm import SigmoidBinarySCM
from mcr.causality.dags import DirectedAcyclicGraph
import numpy as np

savepath = 'scms/'

scm = SigmoidBinarySCM.load(savepath + 'example5_v2/')

context = scm.sample_context(100)
# scm.compute_node('y')
data = scm.compute()

obs = data.iloc[0, :]
obs = obs.loc[obs.index != 'y']

scm_abd = scm.abduct(obs)
smpl = scm_abd.sample_context(10000)
smpl_data = scm_abd.compute()

# hand-engineer the intervention

scm_abd.compute(do={'x1': 1.0}).describe()['y']

scm_true = scm.copy()
scm_true.set_noise_values(smpl.iloc[0, :].to_dict())
scm_true.compute(do={'x1': 1.0, 'x6':1.0}).describe()['y']


import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(smpl['u_y'])
plt.show()

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