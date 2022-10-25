# how to sample data
from mcr.causality.scms.examples import scm_dict
print(scm_dict.keys())

scm = scm_dict['3var-noncausal']
N = 10000

U = scm.sample_context(N)  # exogenous noise
X = scm.compute()  # endogenous variables
