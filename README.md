# Meaningful Causal Recourse: Experiments


## Installation

### Dependencies

The dependencies can be found in the `requirements.txt` file.
For ray, only `ray[tune]` is required. Follow the instructions in the [ray documentation](https://docs.ray.io/en/latest/installation.html).

### Installation with pip

In order to install the package and its functionality, run ``pip install -e mcr`` in the superfolder of the one that contains this README file.

## The Package

The package allows to apply MCR and CR to binomial binary problems.
An SCM can be specified and a series of experiments can be run.

## How to Reproduce Results

In order to generate the examplary SCMs, you can run ```python scripts/generate-scms.py```.
All SCMS that are required to reproduce the paper results have already been generated and can be found in ```scms/```.

In order to reproduce a full comparison of all four methods on a given example, run ```scripts/experiment-all-types.py```.
In order to reproduce the robustness results, run ```scripts/experiemnt-robustness.py```.
The scripts ``scripts/compile-all-types.py`` and ``scripts/compile-robustness.py`` are used to copile the results for the respective experiments.

### Experiment All Types

In order to reproduce our results, run the following script for ``[gamma]`` 0.95 and 0.9

```
python scripts/experiment-all-types.py ../path-to-results-all-types/ 1 2000 [gamma] 0.5 1 --lbd 2 --scm_loadpath scms/example1/ --predict_individualized True
```

Then, when all the experiments that you wanted to run (e.g. for the different gamma levels) are completed, run

```
python scripts/compile-all-types.py ../path-to-results-all-types/
```

In the ``path-to-results-all-types/`` folder you can then find two files called

- ``resultss.csv``: summary statistics for all experiment folders in the specified result folder. mean and standard deviation for
  - `eta_mean`: specified desired acceptance rate
  - `gamma_mean`: specified desired improvement rate
  - `perc_recomm_found`: percent of recourse-seeking individuals for which a recommendation could be made
  - `eta_obs_mean`: average acceptance rate (observed)
  - `gamma_obs_mean`: average improvement rate (observed)
  - `eta_obs_individualized`: average acceptance rate for the individualized post-recourse predictor (observed)
  - `intv-cost_mean`: averge cost of the suggested interventions
  - `[...]_std`: the respective standard deviations
- ``invs_resultss.csv``: overview of interventions performed for each of the variables as well as aggregated for causes and non-causal variables

These values indicate the results presented in the paper.

### Experiment Robustness

In order to reproduce our results, run the following script for ``[gamma]`` levels 0.9 and 0.95

```
python robustness.py ../scms/example1/ [gamma] 2 0.5 1500 ../path-to-results-robustness/

```

In the ``path-to-results-robustness/`` you can find a folder each run, containing files that capture all relevant aspects of the experiment.
They can be compiled using 

```
python compile-robustness.py path-to-results-robustness/
```

Then, in each of the folders there is a file ``aggregated_result.csv``, a table with

- `eta_obs`: The observed acceptance rate on the original model
- `eta_obs_refit`: The observed acceptance rate on the refit
- `r_type`: whether subpopulation-based or individualized recourse was used
- `t_type`: whether improvement or acceptance was targeted

