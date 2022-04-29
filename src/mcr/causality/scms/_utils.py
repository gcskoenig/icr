import numpyro
import pyro
import torch
import numpy as np

def numpyrodist_to_pyrodist(d):
    if isinstance(d, numpyro.distributions.discrete.CategoricalProbs):
        probs = d.probs
        t_probs = torch.tensor(np.array(probs))
        return pyro.distributions.Binomial(probs=t_probs)
    elif isinstance(d, numpyro.distributions.Normal):
        loc = torch.tensor(np.array(d.loc))
        scale = torch.abs(torch.tensor(np.array(d.scale)))
        return pyro.distributions.Normal(loc, scale)
    elif isinstance(d, numpyro.distributions.Uniform):
        high = torch.tensor(np.array(d.low))
        low = torch.tensor(np.array(d.low))
        return pyro.distributions.Uniform(high=high, low=low)
    else:
        raise NotImplementedError('Type not implemented')

