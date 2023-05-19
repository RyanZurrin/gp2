import os
import pickle
from .classifier import Classifier
from gp2.gp2.util import Util


class BaseKerasSegmentationClassifier(Classifier):
    def __init__(self, verbose, workingdir, **kwargs):
        super().__init__(verbose, workingdir, **kwargs)

    def build(self):
        """ Build the model. """
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metric)

    def train(self,
              X_train,
              y_train,
              X_val,
              y_val,
              patience_counter=2,
              batch_size=64,
              epochs=100,
              call_backs=None,
              ):
        """ Train the model.
        Parameters
        ----------
        X_train : numpy.ndarray
            The training images.
        y_train : numpy.ndarray
            The training masks.
        X_val : numpy.ndarray
            The validation images.
        y_val : numpy.ndarray
            The validation masks.
        patience_counter : int
            The number of epochs to wait before early stopping.
        batch_size : int
            The batch size to use.
        epochs : int
            The number of epochs to train for.
        call_backs : list
            The list of callbacks to use.
        """
        from tensorflow.keras import callbacks
        super().train(X_train, y_train, X_val, y_val, patience_counter)
        checkpoint_file = os.path.join(self.workingdir, self.name)
        checkpoint_file = Util.create_numbered_file(checkpoint_file,
                                                    f'{self.name}_model')

        if call_backs is None:
            call_backs = [callbacks.EarlyStopping(patience=patience_counter,
                                                  monitor='loss',
                                                  verbose=0),
                          callbacks.ModelCheckpoint(checkpoint_file,
                                                    save_weights_only=False,
                                                    monitor='val_loss',
                                                    mode='min',
                                                    verbose=0,
                                                    save_best_only=True)]
        else:
            call_backs = call_backs

        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=(X_val, y_val),
                                 callbacks=call_backs,
                                 )

        history_file = os.path.join(self.workingdir, f'{self.name}_history')
        history_file = Util.create_numbered_file(history_file, '.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        print('Model saved to: {}'.format(checkpoint_file))
        print('History saved to: {}'.format(history_file))

        return history

    def predict(self, X_test, y_pred, threshold=0.5):
        """ Predict the masks for the given images.
        Parameters
        ----------
        X_text : numpy.ndarray
            The images to predict the masks for.
        y_pred : numpy.ndarray
            The predicted masks.
        threshold : float
            The threshold to use for the predictions.
        """
        predictions = self.model.predict(X_test)

        predictions[predictions >= threshold] = 1.0
        predictions[predictions < threshold] = 0.0

        scores = self.model.evaluate(X_test, y_pred, verbose=0)

        return predictions, scores

    # print all the classes attributes using vars() function
    def __str__(self):
        return str(vars(self))
