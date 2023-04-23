import os, sys


class Discriminator:

    def __init__(self, verbose=True, workingdir='/tmp', **kwargs):
        self.verbose = verbose
        self.workingdir = workingdir
        pass

    def build(self):
        pass

    def train(self, X_train_images, X_train_masks,
              y_train, X_val_images, X_val_masks, y_val):
        pass

    def predict(self, X_test_images, X_test_masks, y_pred):
        pass
