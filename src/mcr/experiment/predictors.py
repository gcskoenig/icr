import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def get_tuning_grid_rf(*args, **kwargs):
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 5)]
    max_features = ['auto', 'sqrt']
    max_samples = uniform(0, 1)
    max_depth = [int(x) for x in np.linspace(10, 30, num=5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    # bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_samples': max_samples}
    return random_grid


def get_tuning_rf(n_iter, cv, *args, verbose=0, **kwargs):
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=get_tuning_grid_rf(*args, **kwargs), n_iter=n_iter,
                                   cv=cv, verbose=verbose, random_state=random.randint(0, 2**4), n_jobs=-1)
    return rf_random