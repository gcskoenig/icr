import numpyro
import torch
import pyro
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
        low = torch.tensor(np.array(d.low))
        high = torch.tensor(np.array(d.high))
        return pyro.distributions.Uniform(high=high, low=low)
    else:
        raise NotImplementedError('Type not implemented')

def add_uncertainty(p):
    p = p.double()
    if p > 0.5:
        return p - 0.01
    else:
        return p + 0.01