import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperopt import hp, rand, fmin, Trials
from sklearn.model_selection import cross_val_score, train_test_split
from omama import Features
import warnings

warnings.filterwarnings('ignore')


class HyperTuner(object):
    def __init__(self,
                 algorithm,
                 param_space,
                 feature_type,
                 norm_type,
                 data,
                 gt_labels,
                 scoring='accuracy',
                 cv=5,
                 n_jobs=-1,
                 max_evals=100,
                 test_size=0.2,
                 random_state=42,
                 tts_random_state=0,
                 obj_func=None,
                 **kwargs):
        """
        Hyperparameter tuner.

        Parameters
        ----------
        algorithm:
            algorithm to be used for hyperparameter tuning.
        param_space:
            parameter space to be used for hyperparameter tuning.
        feature_type:
            feature type to be used for hyperparameter tuning.
        norm_type:
            normalization type to be used for hyperparameter tuning.
        data:
            data to be used for hyperparameter tuning.
        gt_labels:
            ground truth labels to be used for hyperparameter tuning.
        scoring:
            scoring to be used for hyperparameter tuning.
        cv:
            cross validation to be used for hyperparameter tuning.
        n_jobs:
            number of jobs to be used for hyperparameter tuning.
        max_evals:
            maximum number of evaluations to be used for hyperparameter tuning.
        test_size:
            test size to be used for hyperparameter tuning.
        random_state:
            random state to be used for hyperparameter tuning.
        ts_random_state:
            random state to be used for hyperparameter tuning.
        obj_func:
            custom objective function to be used for hyperparameter tuning.
        **kwargs:
            keyword arguments to be used for hyperparameter tuning.
        """
        self.algorithm = algorithm
        self.param_space = param_space
        self.feature_type = feature_type
        self.norm_type = norm_type
        self.data = data
        self.gt_labels = gt_labels
        self.scoring = scoring
        self.feat_vect = Features.get_features(data,
                                               feature_type,
                                               norm_type)
        self.X_df = pd.DataFrame(self.feat_vect)
        self.y_df = pd.Series(self.gt_labels)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_df, self.y_df, test_size=test_size,
            random_state=tts_random_state)
        self.cv = cv
        self.n_jobs = n_jobs
        self.max_evals = max_evals
        self.test_size = test_size
        self.random_state = random_state
        self.tts_random_state = tts_random_state
        self.obj_func = obj_func
        self.trials = Trials()
        self.best_params = None
        self.kwargs = kwargs

    def __str__(self):
        """
        String representation of the class and all its attributes.

        Returns
        -------
        str
            string representation of the class.
        """
        return str(self._parse_param_space())

    def objective(self, params):
        """
        Objective function for hyperparameter optimization.

        Parameters
        ----------
        params:
            parameters to be used for hyperparameter optimization.

        Returns
        -------
        float
            negative cross validation score.
        """
        clf = self.algorithm(**params)
        clf.fit(self.X_train, self.y_train)

        return -cross_val_score(clf,
                                self.X_train,
                                self.y_train,
                                cv=self.cv,
                                n_jobs=self.n_jobs,
                                scoring=self.scoring).mean()

    def optimize(self, display=True, plot=True, **kwargs):
        """
        Optimize the hyperparameters.

        Parameters
        ----------
        display:
            display the results of the hyperparameter optimization.
        plot:
            plot the results of the hyperparameter optimization.
        **kwargs:
            keyword arguments to be used for hyperparameter optimization.

        """
        # check if there is a new parameter space in kwargs to be used
        param_space = kwargs.get('param_space', self.param_space)

        if self.obj_func is not None:
            fn = self.obj_func
        else:
            fn = self.objective

        # get the best parameters
        self.best_params = fmin(fn=fn,
                                space=param_space,
                                algo=rand.suggest,
                                max_evals=self.max_evals,
                                trials=self.trials,
                                rstate=np.random.default_rng(self.random_state),
                                **self.kwargs)

        if display:
            self.display_best_params()

        # plot the results
        if plot:
            self.plot_results()

    def display_best_params(self):
        """
        Display the best parameters.
        """
        # parse the parameter space
        param_dict = self._parse_param_space()
        # get the best parameters
        best_params = pd.DataFrame(self.trials.vals).iloc[
            self.trials.best_trial['tid']].to_dict()

        print('Best parameters:')
        for key, value in best_params.items():
            try:
                test = float(param_dict[key][0])
            except ValueError:
                value = param_dict[key][int(value)]
            print('{}: {}'.format(key, value))

    def print_param_space(self):
        """
        Print the parameter space to a file or to the console.
        """
        param_dict = self._parse_param_space()
        # print the parameter space
        print('Parameter space:')
        for key, value in param_dict.items():
            print('{}: {}'.format(key, value))

    def plot_results(self):
        """
        Plot the results of the hyperparameter optimization.
        """
        # get the results
        results = pd.concat([
            pd.DataFrame(self.trials.vals),
            pd.DataFrame(self.trials.results)],
            axis=1).sort_values(by='loss', ascending=False).reset_index(
            drop=True)

        # plot the results
        results['loss'].plot()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

    def _parse_param_space(self):
        """
        Parse the parameter space to make it readable

        Returns
        -------
        {'contamination': [0.07, 0.1],
            'n_neighbors':  [1, 50, 1.0],
            'method': ['largest', 'mean', 'median']}

        """
        file = 'temp_param_space.txt'
        # open file
        f = open(file, "w")
        # write only the hyperopt_param to the file in a readable format above
        for s in self.param_space:
            f.write('{}: {}'.format(s, self.param_space[s]['hyperopt_param']))
            f.write('\n')
        # close file
        f.close()

        # open file
        f = open(file, "r")
        # read in the file
        lines = f.readlines()
        # close file
        f.close()

        # create a dictionary to store the key and values
        param_dict = {}
        # loop through the lines
        key = ''
        for line in lines:
            # if line does not start with a number then it is a key
            if line and not line[0].isdigit():
                # split the line with the key at the : and store the key
                key = line.split(':')[0]
                # now loop through and save the lines that are between this key
                # and the next key or the end of the file
                values = []
                for line in lines[lines.index(line) + 1:]:
                    # if line does not start with a number then it is a key
                    if line and not line[0].isdigit():
                        # break out of the loop
                        break
                    else:
                        # split the line with the value at the Literal{ and
                        # store the value
                        if 'Literal' in line:
                            value = line.split('Literal{')[1].split('}')[0]
                            # if the value is the same as the key then ignore it
                            if value != key and value != 'hyperopt_param':
                                # store the value in the values list
                                values.append(value)
                            # check the second value in the list and see if it can
                            # be converted to a float and if not then check the
                            # first value and if it can then pop the first value
                            # from the list
                            if len(values) > 1:
                                try:
                                    float(values[1])
                                except ValueError:
                                    try:
                                        int(values[0])
                                        values.pop(0)
                                    except ValueError:
                                        pass

                # store the key and values in the dictionary
                param_dict[key] = values

        # remove the temp file
        os.remove(file)

        return param_dict
