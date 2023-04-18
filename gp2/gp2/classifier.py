import os, sys


class Classifier:

    def __init__(self, verbose=True, workingdir='/tmp', **kwargs):
        self.verbose = verbose
        self.workingdir = workingdir

        pass

    def build(self, **kwargs):
        pass

    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        pass

    def predict(self, X_test, y_pred, threshold):
        pass
