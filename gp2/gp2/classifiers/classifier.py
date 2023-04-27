from abc import ABC, abstractmethod


class Classifier(ABC):
    """
    Base class for classifiers
    """

    def __init__(self, verbose=True, workingdir='/tmp', **kwargs):
        """
        Args:
            verbose: (bool) print verbose output
            workingdir: (str) working directory
            **kwargs: (dict) keyword arguments
        """
        self.verbose = verbose
        self.workingdir = workingdir
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, patience_counter):
        pass

    @abstractmethod
    def predict(self, X_test, y_pred, threshold):
        pass
