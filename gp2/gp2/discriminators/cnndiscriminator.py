from .discriminator import Discriminator


import numpy as np
import os
import pickle

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, \
    LeakyReLU, Dropout, Flatten, Dense

from gp2.gp2.util import Util


class CNNDiscriminator(Discriminator):

    def __init__(self, verbose=True, workingdir='/tmp'):
        super().__init__(verbose, workingdir)

        self.model = self.build()

    def create_convolution_layers(self, input_img, input_shape):
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

        # print('y_train')
        # print(y_train.shape)
        # print(y_train[0:10])

        # print('y_val')
        # print(y_val.shape)
        # print(y_val[0:10])

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