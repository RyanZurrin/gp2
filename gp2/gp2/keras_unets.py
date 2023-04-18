import os
import pickle

from keras import losses, metrics
from tensorflow.keras import callbacks, optimizers

from .classifier import Classifier
from .util import Util
from keras_unet_collection import models, base, utils
import tensorflow as tf
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


class KATTUnet2D(Classifier):

    def __init__(self,
                 input_shape=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 stack_num_down=2,
                 stack_num_up=2,
                 activation='ReLU',
                 atten_activation='ReLU',
                 attention='add',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=False,
                 unpool=False,
                 backbone=None,
                 weights=None,
                 freeze_backbone=True,
                 freeze_batch_norm=True,
                 name='attunet',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        super().__init__(verbose=verbose, workingdir=workingdir)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        if optimizer is None:
            self.optimizer = optimizers.Adam(lr=1e-4)

        if loss is None:
            self.loss = losses.binary_crossentropy

        if metric is None:
            self.metric = [dice_coef]

        if filter_num is None:
            filter_num = [32, 64, 128, 256, 512]
        self.input_shape = input_shape
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.atten_activation = atten_activation
        self.attention = attention
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.backbone = backbone
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        self.name = name
        self.model = models.att_unet_2d(self.input_shape,
                                        filter_num=self.filter_num,
                                        n_labels=self.n_labels,
                                        stack_num_down=self.stack_num_down,
                                        stack_num_up=self.stack_num_up,
                                        activation=self.activation,
                                        atten_activation=self.atten_activation,
                                        attention=self.attention,
                                        output_activation=self.output_activation,
                                        batch_norm=self.batch_norm,
                                        pool=self.pool, unpool=self.unpool,
                                        backbone=self.backbone,
                                        weights=self.weights,
                                        freeze_backbone=self.freeze_backbone,
                                        freeze_batch_norm=self.freeze_batch_norm,
                                        name=self.name)
        self.build()

        if self.verbose:
            self.model.summary()

    def build(self):
        """ Build the model.
        Returns
        -------
        model : keras.models.Model
            The compiled model.

        """
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
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
              **kwargs):
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
        checkpoint_file = os.path.join(self.workingdir, 'kattunet')
        checkpoint_file = Util.create_numbered_file(checkpoint_file,
                                                    'kattunet_model')

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
                                 **kwargs)

        history_file = os.path.join(self.workingdir, 'attunet_history')
        history_file = Util.create_numbered_file(history_file, '.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        print('Model saved to: {}'.format(checkpoint_file))
        print('History saved to: {}'.format(history_file))

        return history

    def predict(self, X_test, y_pred, threshold=0.5):
        """ Predict the masks for the images.
        Parameters
        ----------
        X_test : numpy.ndarray
            The test images.
        y_pred : numpy.ndarray
            The predicted masks.
        threshold : float
            The threshold to use for the masks.
        Returns
        -------
        y_pred : numpy.ndarray
            The predicted masks.
        """
        predictions = self.model.predict(X_test)

        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        scores = self.model.evaluate(X_test, y_pred, verbose=0)

        return predictions, scores


class KUNet2D(Classifier):

    def __init__(self,
                 input_shape=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 stack_num_down=2,
                 stack_num_up=2,
                 activation='ReLU',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=False,
                 unpool=False,
                 backbone=None,
                 weights=None,
                 freeze_backbone=True,
                 freeze_batch_norm=True,
                 name='unet',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        super().__init__(verbose=verbose, workingdir=workingdir)

        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        if optimizer is None:
            self.optimizer = optimizers.Adam(lr=1e-4)

        if loss is None:
            self.loss = losses.binary_crossentropy

        if metric is None:
            self.metric = [dice_coef]

        if filter_num is None:
            filter_num = [32, 64, 128, 256, 512]
        self.input_shape = input_shape
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.backbone = backbone
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        self.name = name
        self.model = models.unet_2d(self.input_shape,
                                    filter_num=self.filter_num,
                                    n_labels=self.n_labels,
                                    stack_num_down=self.stack_num_down,
                                    stack_num_up=self.stack_num_up,
                                    activation=self.activation,
                                    output_activation=self.output_activation,
                                    batch_norm=self.batch_norm,
                                    pool=self.pool, unpool=self.unpool,
                                    backbone=self.backbone,
                                    weights=self.weights,
                                    freeze_backbone=self.freeze_backbone,
                                    freeze_batch_norm=self.freeze_batch_norm,
                                    name=self.name)

        self.build()

        if verbose:
            self.model.summary()

    def build(self):
        """ Build the model."""
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
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
              **kwargs):
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
        checkpoint_file = os.path.join(self.workingdir, 'kunet2d')
        checkpoint_file = Util.create_numbered_file(checkpoint_file,
                                                    'kunet2d_model')

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
                                 **kwargs)

        history_file = os.path.join(self.workingdir, 'kunet2d_history')
        history_file = Util.create_numbered_file(history_file, '.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        print('Model saved to: {}'.format(checkpoint_file))
        print('History saved to: {}'.format(history_file))

        return history

    def predict(self, X_test, y_pred, threshold=0.5):
        """ Predict the masks for the images.
        Parameters
        ----------
        X_test : numpy.ndarray
            The test images.
        y_pred : numpy.ndarray
            The predicted masks.
        threshold : float
            The threshold to use for the masks.
        Returns
        -------
        y_pred : numpy.ndarray
            The predicted masks.
        """
        predictions = self.model.predict(X_test)

        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        scores = self.model.evaluate(X_test, y_pred, verbose=0)

        return predictions, scores


class KUNetPlus2D(Classifier):

    def __init__(self,
                 input_shape=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 stack_num_down=2,
                 stack_num_up=2,
                 activation='ReLU',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=False,
                 unpool=False,
                 backbone=None,
                 weights=None,
                 freeze_backbone=True,
                 freeze_batch_norm=True,
                 name='unet',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        super().__init__(verbose=verbose, workingdir=workingdir)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        if optimizer is None:
            self.optimizer = optimizers.Adam(lr=1e-4)

        if loss is None:
            self.loss = losses.binary_crossentropy

        if metric is None:
            self.metric = [dice_coef]

        if filter_num is None:
            filter_num = [32, 64, 128, 256, 512]
        self.input_shape = input_shape
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.backbone = backbone
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        self.name = name
        self.model = models.unet_plus_2d(self.input_shape,
                                         filter_num=self.filter_num,
                                         n_labels=self.n_labels,
                                         stack_num_down=self.stack_num_down,
                                         stack_num_up=self.stack_num_up,
                                         activation=self.activation,
                                         output_activation=self.output_activation,
                                         batch_norm=self.batch_norm,
                                         pool=self.pool, unpool=self.unpool,
                                         backbone=self.backbone,
                                         weights=self.weights,
                                         freeze_backbone=self.freeze_backbone,
                                         freeze_batch_norm=self.freeze_batch_norm,
                                         name=self.name)
        self.build()

        if verbose:
            self.model.summary()

    def build(self):
        """ Build the model. """
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
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
              **kwargs):
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
        checkpoint_file = os.path.join(self.workingdir, 'kunetplus2d')
        checkpoint_file = Util.create_numbered_file(checkpoint_file,
                                                    'kunetplus2d_model')

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
                                 **kwargs)

        history_file = os.path.join(self.workingdir, 'kunetplus2d_history')
        history_file = Util.create_numbered_file(history_file, '.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        print('Model saved to: {}'.format(checkpoint_file))
        print('History saved to: {}'.format(history_file))

        return history

    def predict(self, X_test, y_pred, threshold=0.5):
        """ Predict the masks for the images.
            Parameters
            ----------
            X_test : numpy.ndarray
                The test images.
            y_pred : numpy.ndarray
                The predicted masks.
            threshold : float
                The threshold to use for the masks.
            Returns
            -------
            y_pred : numpy.ndarray
                The predicted masks.
            """
        predictions = self.model.predict(X_test)

        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        scores = self.model.evaluate(X_test, y_pred, verbose=0)

        return predictions, scores


class KResUNet2D(Classifier):

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 dilation_num=1,
                 n_labels=1,
                 aspp_num_down=256,
                 aspp_num_up=128,
                 activation='ReLU',
                 output_activation='Softmax',
                 batch_norm=True,
                 pool=True,
                 unpool=True,
                 name='resunet',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        super().__init__(verbose=verbose, workingdir=workingdir)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        if filter_num is None:
            filter_num = [32, 64, 128, 256, 512]

        if optimizer is None:
            self.optimizer = optimizers.Adam(lr=1e-4)

        if loss is None:
            self.loss = losses.binary_crossentropy

        if metric is None:
            self.metric = [metrics.binary_accuracy]

        self.input_shape = input_size
        self.filter_num = filter_num
        self.dilation_num = dilation_num
        self.n_labels = n_labels
        self.aspp_num_down = aspp_num_down
        self.aspp_num_up = aspp_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.name = name
        self.model = models.resunet_a_2d(self.input_shape,
                                         filter_num=self.filter_num,
                                         dilation_num=self.dilation_num,
                                         n_labels=self.n_labels,
                                         aspp_num_down=self.aspp_num_down,
                                         aspp_num_up=self.aspp_num_up,
                                         activation=self.activation,
                                         output_activation=self.output_activation,
                                         batch_norm=self.batch_norm,
                                         pool=self.pool, unpool=self.unpool,
                                         name=self.name)

        self.build()

        if verbose:
            self.model.summary()

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
              **kwargs):
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
        checkpoint_file = os.path.join(self.workingdir, 'resunet')
        checkpoint_file = Util.create_numbered_file(checkpoint_file,
                                                    'resunet_model')

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
                                 **kwargs)

        history_file = os.path.join(self.workingdir, 'resunet_history')
        history_file = Util.create_numbered_file(history_file, '.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        print('Model saved to: {}'.format(checkpoint_file))
        print('History saved to: {}'.format(history_file))

        return history

    def predict(self, X_test, y_pred, threshold=0.5):
        """ Predict the masks for the images.
        Parameters
        ----------
        X_test : numpy.ndarray
            The test images.
        y_pred : numpy.ndarray
            The predicted masks.
        threshold : float
            The threshold to use for the masks.
        Returns
        -------
        y_pred : numpy.ndarray
            The predicted masks.
        """
        predictions = self.model.predict(X_test)

        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        scores = self.model.evaluate(X_test, y_pred, verbose=0)

        return predictions, scores


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
        (1 - dice_coef(y_true, y_pred))
