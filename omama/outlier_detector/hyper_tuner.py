import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperopt import hp, rand, fmin, Trials
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from omama.feature_extractor import Features


class HyperTune(object):
    def __init__(self,
                 algorithm,
                 feature_type,
                 norm_type,
                 param_space,
                 X,
                 y,
                 cv=5,
                 n_iter=100,
                 max_evals=1000,
                 test_size=0.2,
                 random_state=42,
                 **kwargs):
        self.algorithm = algorithm
        self.param_space = param_space
        self.feature_type = feature_type
        self.norm_type = norm_type
        self.X = X
        self.y = y
        self.cv = cv
        self.n_iter = n_iter
        self.max_evals = max_evals
        self.test_size = test_size
        self.random_state = random_state
        self.trials = Trials()
        self.kwargs = kwargs

    def objective(self, params):
        """
        Objective function for hyperparameter optimization.
        """
        # get the algorithm
        algo = self.algorithm(**params)

        # get the features
        features = Features.get_features(self.X,
                                         feature_type=self.feature_type,
                                         norm_type=self.norm_type)

        # get the scores
        scores = cross_val_score(algo,
                                 features,
                                 self.y,
                                 cv=self.cv,
                                 n_jobs=-1)

        # return the mean score
        return -np.mean(scores)

    def optimize(self):
        """
        Optimize the hyperparameters.
        """
        # get the best parameters
        best = fmin(self.objective,
                    self.param_space,
                    algo=rand.suggest,
                    max_evals=self.max_evals,
                    trials=self.trials,
                    rstate=np.random.RandomState(self.random_state))

        # get the best score
        best_score = -self.trials.best_trial['result']['loss']

        # get the best parameters
        best_params = self.trials.best_trial['misc']['vals']

        # return the best parameters and the best score
        return best_params, best_score
