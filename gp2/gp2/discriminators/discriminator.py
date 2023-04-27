from abc import ABC, abstractmethod


class Discriminator(ABC):

    def __init__(self, verbose=True, workingdir='/tmp', **kwargs):
        self.verbose = verbose
        self.workingdir = workingdir
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, X_train_images, X_train_masks,
              y_train, X_val_images, X_val_masks, y_val):
        pass

    @abstractmethod
    def predict(self, X_test_images, X_test_masks, y_pred):
        pass