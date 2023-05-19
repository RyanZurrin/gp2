from .discriminator import Discriminator
import os
import pickle
from gp2.gp2.util import Util

class CNNDiscriminator(Discriminator):

    def __init__(self, verbose=True, workingdir='/tmp'):
        super().__init__(verbose, workingdir)

        self.model = self.build()

    def create_convolution_layers(self, input_img, input_shape):
        """ Create the convolution layers

        Args:
            input_img:  (tf.keras.layers.Input) input layer
            input_shape:  (tuple) input shape

        Returns:
        """
        from tensorflow.keras.layers import Conv2D,  MaxPooling2D, LeakyReLU, Dropout
        # 512
        model = Conv2D(16, (3, 3), padding='same', input_shape=input_shape)(
            input_img)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D((2, 2), padding='same')(model)
        model = Dropout(0.25)(model)
        # 256

        model = Conv2D(32, (3, 3), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D((2, 2), padding='same')(model)
        model = Dropout(0.25)(model)
        # 128

        model = Conv2D(64, (3, 3), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2), padding='same')(model)
        model = Dropout(0.25)(model)
        # 64

        model = Conv2D(128, (3, 3), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2), padding='same')(model)
        model = Dropout(0.4)(model)
        # 32

        model = Conv2D(256, (3, 3), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2), padding='same')(model)
        model = Dropout(0.4)(model)
        # 16

        return model

    def build(self, image_shape=(512, 512, 1), num_classes=2):
        """ Builds the model.

        :param image_shape: The shape of the input images.
        :param num_classes: The number of classes.
        :return: The model.
        """
        from tensorflow.keras.layers import Input, concatenate,  LeakyReLU, \
            Dropout, Flatten, Dense
        from tensorflow.keras.models import Model
        super().build()

        image_input = Input(shape=image_shape)
        image_model = self.create_convolution_layers(image_input, image_shape)

        mask_input = Input(shape=image_shape)
        mask_model = self.create_convolution_layers(mask_input, image_shape)

        conv = concatenate([image_model, mask_model])

        conv = Flatten()(conv)

        dense = Dense(512)(conv)
        dense = LeakyReLU(alpha=0.1)(dense)
        dense = Dropout(0.5)(dense)

        output = Dense(num_classes, activation='softmax')(dense)

        model = Model(inputs=[image_input, mask_input], outputs=[output])

        # compile the model
        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, X_train_images, X_train_masks, y_train,
              X_val_images, X_val_masks, y_val,
              patience_counter=10, batch_size=64, epochs=100):
        """ Trains the model.

        :param X_train_images: The training images.
        :param X_train_masks: The training masks.
        :param y_train: The training labels.
        :param X_val_images: The validation images.
        :param X_val_masks: The validation masks.
        :param y_val: The validation labels.
        :param patience_counter: The patience counter for early stopping.
        :param batch_size: The batch size.
        :param epochs: The number of epochs.
        :return:
        """
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        super().train(X_train_images, X_train_masks, y_train,
                      X_val_images, X_val_masks, y_val)

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
        """ Predicts the labels for the given test data.

        :param X_test_images: The test images.
        :param X_test_masks: The test masks.
        :param y_test: The test labels.
        :return: The predicted labels and the scores.
        """
        import numpy as np
        from tensorflow.keras.utils import to_categorical
        predictions = self.model.predict(x=[X_test_images, X_test_masks])

        # grab the most likely label from the categorical representation
        predictions = np.argmax(predictions, axis=-1)

        y_test = to_categorical(y_test)

        scores = self.model.evaluate(x=[X_test_images, X_test_masks], y=y_test)

        return predictions, scores