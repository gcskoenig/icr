from mcr.experiment.run import run_experiment
from mcr.experiment.compile import compile_experiments
import random
import os

# savepath = '../experiments/remote-experiments/test_generic/greenfield/'
savepath = '/home/gcsk/data/mcr-experiments/test_generic/3var-noncausal/'

id = random.randint(0, 2**10)
print(id)

os.mkdir(savepath + f"_{id}/")

run_experiment('3var-noncausal', 2000, 0.9, 0.5, 10, savepath + f'_{id}',
               NGEN=100, assess_robustness=False, iterations=5)
compile_experiments(savepath, assess_robustness=False, scm_name='3var-noncausal')


savepath = '/home/gcsk/data/experiments/mcr-experiments/test_generic/3var-causal/'

id = random.randint(0, 2**10)
print(id)

os.mkdir(savepath + f"_{id}/")

run_experiment('3var-causal', 2000, 0.9, 0.5, 10, savepath + f'_{id}',
               NGEN=100, assess_robustness=False, iterations=5)
compile_experiments(savepath, assess_robustness=False, scm_name='3var-causal')