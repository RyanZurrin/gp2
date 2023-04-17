# keras unet class for adding support for keras unet
import os
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, \
    callbacks
from .classifier import Classifier
from .util import Util


class UNetPLUS(Classifier):

    def __init__(self,
                 input_shape=(512, 512, 1),
                 num_classes=1,
                 base_filters=4,
                 dropout=0.1,
                 batchnorm=True,
                 activation='relu',
                 kernel_size=3,
                 padding='same',
                 strides=1,
                 depth=4,
                 num_layers=4,
                 optimizer=None,
                 loss=None,
                 _metrics=None,
                 verbose=True,
                 workingdir='/tmp'):
        """ Initialize the KerasUNet class.

            Parameters
            ----------
            input_shape : tuple
                The shape of the input image.
            num_classes : int
                The number of classes to predict.
            base_filters : int
                The number of filters to use in the convolutional layers.
            dropout : float
                The dropout rate.
            batchnorm : bool
                Whether to use batch normalization.
            activation : str
                The activation function to use.
            kernel_size : int
                The kernel size to use in the convolutional layers.
            padding : str
                The padding to use in the convolutional layers.
            strides : int
                The strides to use in the convolutional layers.
            depth : int
                The depth of the network.
            num_layers : int
                The number of layers to use in the network.
            optimizer : keras.Optimizers
                The optimizer to use.
            loss : keras.Losses
                The loss function to use.
            _metrics : list
                The metrics to use.
            verbose : bool
                Whether to print the model summary.
            workingdir : str
                The working directory to use for saving the model.

            """
        super().__init__(verbose=verbose, workingdir=workingdir)

        if optimizer is None:
            self._optimizer = optimizers.Adam(lr=1e-4)

        if loss is None:
            self._loss = losses.binary_crossentropy

        if _metrics is None:
            self._metrics = [UNetPLUS.dice_coef]

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.depth = depth
        self.num_layers = num_layers

        self.model = self.build()

        if verbose:
            self.model.summary()

    def build(self):
        """ Build the model.

        Returns
        -------
        model : keras.Model
            The model.
        """
        super().build()

        inputs = layers.Input(shape=self.input_shape)

        # Contracting path (encoder)
        x = inputs
        skips = []
        for i in range(self.depth):
            for j in range(self.num_layers):
                x = layers.Conv2D(self.base_filters * (2 ** i),
                                  self.kernel_size,
                                  padding=self.padding,
                                  strides=self.strides,
                                  activation=self.activation)(x)
                if self.batchnorm:
                    x = layers.BatchNormalization()(x)
                if self.dropout:
                    x = layers.Dropout(self.dropout)(x)
            skips.append(x)
            x = layers.MaxPooling2D((2, 2))(x)

        # Bottleneck (encoder)
        for i in range(self.num_layers):
            x = layers.Conv2D(self.base_filters * (2 ** self.depth),
                              self.kernel_size,
                              padding=self.padding,
                              strides=self.strides,
                              activation=self.activation)(x)
            if self.batchnorm:
                x = layers.BatchNormalization()(x)
            if self.dropout:
                x = layers.Dropout(self.dropout)(x)

        # Expansive path (decoder)
        for i in reversed(range(self.depth)):
            x = layers.Conv2DTranspose(self.base_filters * (2 ** i),
                                       (2, 2),
                                       strides=(2, 2),
                                       padding='same')(x)
            x = layers.concatenate([x, skips[i]])
            for j in range(self.num_layers):
                x = layers.Conv2D(self.base_filters * (2 ** i),
                                  self.kernel_size,
                                  padding=self.padding,
                                  strides=self.strides,
                                  activation=self.activation)(x)
                if self.batchnorm:
                    x = layers.BatchNormalization()(x)
                if self.dropout:
                    x = layers.Dropout(self.dropout)(x)

        # Output layer
        x = layers.Conv2D(self.num_classes,
                          self.kernel_size,
                          padding=self.padding,
                          strides=self.strides,
                          activation='sigmoid')(x)

        out = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(x)

        model = models.Model(inputs=[inputs], outputs=[out])

        model.compile(optimizer=self._optimizer,
                      loss=self._loss,
                      metrics=self._metrics)

        return model

    def train(self,
              X_train,
              y_train,
              X_val,
              y_val,
              patience_counter=2,
              batch_size=64,
              epochs=100,
              call_backs=None):
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
        super().train(X_train, y_train, X_val, y_val)

        # create the checkpoint files
        checkpoint_file = os.path.join(self.workingdir, 'kunet')
        checkpoint_file = Util.create_numbered_file(checkpoint_file, 'model')

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

        # get the history
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=call_backs,
                                 validation_data=(X_val, y_val),
                                 verbose=int(self.verbose))

        # save the history
        history_file = os.path.join(self.workingdir, 'khistory')
        history_file = Util.create_numbered_file(history_file, '.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        print('Model saved to: {}'.format(checkpoint_file))
        print('History saved to: {}'.format(history_file))

        return history

    def predict(self, X_text, y_pred, threshold=0.5):
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
        predictions = self.model.predict(X_text)

        # threshold the predictions
        predictions[predictions >= threshold] = 1
        predictions[predictions < (1 - threshold)] = 0

        # get the predicted masks
        scores = self.model.evaluate(X_text, y_pred, verbose=0)

        return predictions, scores

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1e-9):
        """ Calculate the dice coefficient.
        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.
        smooth : float
            The smoothing factor.

        Returns
        -------
        float
            The dice coefficient.
        """
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        """ Calculate the loss.
        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.

        Returns
        -------
        float
            The loss.
        """
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + \
            (1 - UNetPLUS.dice_coef(y_true, y_pred))
