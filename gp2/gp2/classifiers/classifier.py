from abc import ABC, abstractmethod


class Classifier(ABC):
    """ Abstract Base class for classifiers

    """

    def __init__(self, verbose=True, workingdir='/tmp', **kwargs):
        """ Initialize the Classifier class.

        Parameters
        ----------
        verbose : bool
            Whether to print the model summary.
        workingdir : str
            The working directory to use for saving the model.
        **kwargs : dict
            Additional keyword arguments.

        """
        self.verbose = verbose
        self.workingdir = workingdir

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, patience_counter):
        pass

    @abstractmethod
    def predict(self, X_test, y_pred, threshold):
        pass
