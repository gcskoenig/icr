# Meaningful Causal Recourse: Experiments


## Installation

We used python version ``3.9.7``. We recommend to set up a fresh virtual environment, to install the dependenciees and then the package.

### Dependencies

We recommend to install ``torch``, ``scikit-learn``, ``matplotlib``, ``seaborn``,  ``ray`` and ``pyro`` seperately.
For torch, follow the instruction on their [website](https://pytorch.org/get-started/locally/).
For scikit-learn check out [their website](https://scikit-learn.org/stable/install.html).
For pyro, follow the instructions on their [website](https://docs.pyro.ai/en/0.3.1/installation.html).
For ray, only `ray[tune]` is required. Follow the instructions in the [ray documentation](https://docs.ray.io/en/latest/installation.html).

The dependencies can be found in the `requirements.txt` file.
You can install them using ``pip install -r requirements.txt``.

### Installation with pip

In order to install the package and its functionality, run ``pip install -e mcr`` in the superfolder of the one that contains this README file.

## The Package

The package allows to apply MCR and CR to binomial binary problems.
An SCM can be specified and a series of experiments can be run.

## How to Reproduce Results

In order to generate the examplary SCMs, you can run ```python scripts/generate-scms.py```.
All SCMS that are required to reproduce the paper results have already been generated and can be found in ```scms/```.

In order to reproduce a full comparison of all four methods on a given example, run ```scripts/experiment-all-types-v2.py```.
After completion, the script automatically calls ```compile.py```, which aggregates the experiment results in several ```.csv``` files.

In order to reproduce our results, run the following script for ``[gamma]`` 0.95 and 0.9 and for ```[scm-path]``` being ```[path-to-readme-folder]/scms/example1/``` (and ```[path-to-readme-folder]/scms/example1_two_unrelated/``` for the model multiplicity results in the appendix).

```
python scripts/experiment-all-types.py [path-to-experiment-folder] [scm-path] [gamma] 6000 5
```

6000 is the overall sample size and 5 is the number of experiment iterations.

In the ``[path-to-experiment-folder]`` folder you can then find two files called

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

These values indicate the results presented in the paper.

The experimental results presented in the paper can be found in the archive `[path-to-readme-folder]/results.zip`.
