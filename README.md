# Improvement-Focused Causal Recourse: Experiments


## Installation

We used python version ``3.9.7``. We recommend to set up a fresh virtual environment, to install the dependenciees and then the package.

### Dependencies

We recommend to install ``torch``, ``scikit-learn``, ``matplotlib``, ``seaborn``,  ``networkx``, ``pyro``, ``deap`` as well as ``jax``and ``numpyro`` seperately.
For torch, follow the instruction on their [website](https://pytorch.org/get-started/locally/).
For scikit-learn check out [their website](https://scikit-learn.org/stable/install.html).
For pyro, follow the instructions on their [website](https://docs.pyro.ai/en/0.3.1/installation.html).

The remaining dependencies can be found in the `requirements.txt` file.
You can install them using ``pip install -r requirements.txt``.

### Installation with pip

In order to install the package and its functionality, run ``pip install -e icr`` in the superfolder of the one that contains this README file.

## The Package

The package allows to apply ICR, CR and CE to problems with specified causal knowledge.
An SCM can be specified and a series of experiments can be run.

## How to Reproduce Results

In order to reproduce our results, run the following script for ``[confidence]`` 0.75, 0.85, 0.95 and 0.9, for ```[savepath]``` being the path were you would like to store the experiment results, and for ```[nr_runs]``` being the number of runs computed for the given configuration.

```bash
python scripts/run_experiments.py 3var-noncausal 4000 200 [confidence] 300 [savepath]/3var-nc/ [nr_runs] --NGEN 600 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True

python scripts/run_experiments.py 3var-causal 4000 200 [confidence] 300 [savepath]/3var-c/ [nr_runs] --NGEN 600 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True

python scripts/run_experiments.py 5var-skill 4000 200 [confidence] 300 [savepath]/5var-skill/ [nr_runs] --NGEN 1000 --POP_SIZE 500 --n_digits 0 --nr_refits 5 --predict_individualized True --model_type rf

python scripts/run_experiments.py 7var-covid 20000 200 [confidence] 2999 [savepath]/7var-covid/ [nr_runs] --NGEN 700 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf
```

The experiments can be compiled and combined into a plot using 

```
python scripts/plots.py --savepath [savepath]
```

In the ``[savepath]`` folder, for each scm you can then find two files called

- ``resultss.csv``: summary statistics for all experiment folders in the specified result folder. mean and standard deviation for
  - `eta_mean`: specified desired acceptance rate
  - `gamma_mean`: specified desired improvement rate
  - `perc_recomm_found`: percent of recourse-seeking individuals for which a recommendation could be made
  - `eta_obs_mean`: average acceptance rate (observed)
  - `gamma_obs_mean`: average improvement rate (observed)
  - `eta_obs_individualized_mean`: average acceptance rate for the individualized post-recourse predictor (observed)
  - `eta_obs_refits_batch0_mean_mean`: average acceptance rate mean over all (model multiplicity) refits on batch 1 evaluated over batch 2. 
  - `intv-cost_mean`: averge cost of the suggested interventions
  - `[...]_std`: the respective standard deviations
- ``invs_resultss.csv``: overview of interventions performed for each of the variables as well as aggregated for causes and non-causal variables

Furthermore you find summary plots and text files that were used to produce the latex tables in the Appendix of the paper.
