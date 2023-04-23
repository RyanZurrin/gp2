from .discriminator import Discriminator
from ..util import Util
from abc import ABC, abstractmethod

import numpy as np
import os
import pickle

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class BaseCNNDiscriminator(Discriminator, ABC):
    """
    Base class for CNN discriminators
    """

    def __init__(self, workingdir='/tmp', verbose=True):
        super().__init__(workingdir)
        self.verbose = verbose

    @abstractmethod
    def create_convolution_layers(self, input_img, input_shape):
        pass

    @abstractmethod
    def build(self):
        pass

    def train(self, X_train_images, X_train_masks, y_train,
              X_val_images, X_val_masks, y_val,
              patience_counter=10, batch_size=64, epochs=100):
        checkpoint_file = os.path.join(self.workingdir, 'cnnd')
        checkpoint_file = Util.create_numbered_file(checkpoint_file, '.model')

        callbacks = [EarlyStopping(patience=patience_counter,
                                   monitor='loss',
                                   verbose=0),
                     ModelCheckpoint(filepath=checkpoint_file,
                                     save_weights_only=False,
                                     monitor='val_loss',
                                     mode='min',
                                     verbose=0,
                                     save_best_only=True)]

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        history = self.model.fit(x=[X_train_images, X_train_masks], y=y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=(
                                     [X_val_images, X_val_masks], y_val),
                                 verbose=0)

        history_file = os.path.join(self.workingdir, 'cnnd_history')
        history_file = Util.create_numbered_file(history_file, '.pickle')

        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        print('Model saved to', checkpoint_file)
        print('History saved to', history_file)

    def predict(self, X_test_images, X_test_masks, y_test):
        predictions = self.model.predict(x=[X_test_images, X_test_masks])

        # grab the most likely label from the categorical representation
        predictions = np.argmax(predictions, axis=-1)

        y_test = to_categorical(y_test)

        scores = self.model.evaluate(x=[X_test_images, X_test_masks], y=y_test)

        return predictions, scores
