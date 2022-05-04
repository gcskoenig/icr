from mcr.experiment.run import run_experiment
from mcr.experiment.compile import compile_experiments
import random
import os

N = 200
gamma = 0.9
thresh = 0.5
lbd = 10
NGEN = 200
POP_SIZE = 500
iterations = 3
n_digits = 2


savepath = '../experiments/remote-experiments/test_generic/greenfield/'
# savepath = '/home/gcsk/data/mcr-experiments/test_generic/3var-noncausal/'

id = random.randint(0, 2**10)
print(id)

os.mkdir(savepath + f"_{id}/")

run_experiment('3var-noncausal', N, gamma, thresh, lbd, savepath + f'_{id}',
               NGEN=NGEN, assess_robustness=False, iterations=iterations, POP_SIZE=POP_SIZE,
               rounding_digits=n_digits)
compile_experiments(savepath, assess_robustness=False, scm_name='3var-noncausal')


savepath = '/home/gcsk/data/experiments/mcr-experiments/test_generic/3var-causal/'

id = random.randint(0, 2**10)
print(id)

os.mkdir(savepath + f"_{id}/")

run_experiment('3var-causal', N, gamma, thresh, lbd, savepath + f'_{id}',
               NGEN=NGEN, POP_SIZE=POP_SIZE, assess_robustness=False, iterations=iterations,
               rounding_digits=n_digits)
compile_experiments(savepath, assess_robustness=False, scm_name='3var-causal')