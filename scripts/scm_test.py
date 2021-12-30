from rfi.backend.causality.scm import LinearGaussianNoiseSCM
from rfi.backend.causality.dags import DirectedAcyclicGraph
from rfi.examples import SyntheticExample

import numpy as np

sigma_medium = 0.5
sigma_low = 0.1

ii_audit = SyntheticExample(
    name='ii-model-audit',
    sem=LinearGaussianNoiseSCM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 1, 1],
                                       [0, 0, 0],
                                       [0, 0, 0]]),
            var_names=['x1', 'x2', 'y']
        ),
        coeff_dict={'x2': {'x1': 1.0}, 'y': {'x1': 1.0}},
        noise_std_dict={'x1': sigma_medium, 'x2': sigma_medium, 'y': sigma_low}
    )
)

scm = ii_audit.sem

scm.sample_context(100)
data = scm.compute()

obs = data.iloc[0, slice(None)]
scm_obs = scm.abduct(obs)
scm_obs.sample_context(100)
scm_obs.compute()

obs_sub = obs[obs.index != 'y']
scm_obs_sub = scm.abduct(obs_sub)
scm_obs_sub.sample_context(100)
sample = scm_obs_sub.compute(do={'x1': 15})

