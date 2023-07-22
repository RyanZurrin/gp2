from .classifier import Classifier
from gp2.gp2.util import Util

import os
import pickle


class UNet(Classifier):

    def __init__(self, verbose=True, workingdir='/tmp'):
        super().__init__(verbose, workingdir)

        print('*** GP2  Unet ***')
        print('Working directory:', self.workingdir)

        self.model = self.build()

    def build(self, input_shape=(512, 512, 1), num_classes=1):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, concatenate, Conv2D, \
            MaxPooling2D,  Activation, UpSampling2D, BatchNormalization
        from tensorflow.keras.optimizers import RMSprop
        super().build()

        inputs = Input(shape=input_shape)
        # 512

        encoder1 = Conv2D(16, (3, 3), padding='same', input_shape=input_shape)(
            inputs)
        encoder1 = BatchNormalization()(encoder1)
        encoder1 = Activation('relu')(encoder1)
        encoder1 = Conv2D(16, (3, 3), padding='same')(encoder1)
        encoder1 = BatchNormalization()(encoder1)
        encoder1 = Activation('relu')(encoder1)
        encoder1_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder1)
        # 256

        encoder2 = Conv2D(32, (3, 3), padding='same')(encoder1_layer)
        encoder2 = BatchNormalization()(encoder2)
        encoder2 = Activation('relu')(encoder2)
        encoder2 = Conv2D(32, (3, 3), padding='same')(encoder2)
        encoder2 = BatchNormalization()(encoder2)
        encoder2 = Activation('relu')(encoder2)
        encoder2_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder2)
        # 128

        encoder3 = Conv2D(64, (3, 3), padding='same')(encoder2_layer)
        encoder3 = BatchNormalization()(encoder3)
        encoder3 = Activation('relu')(encoder3)
        encoder3 = Conv2D(64, (3, 3), padding='same')(encoder3)
        encoder3 = BatchNormalization()(encoder3)
        encoder3 = Activation('relu')(encoder3)
        encoder3_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder3)
        # 64

        encoder4 = Conv2D(128, (3, 3), padding='same')(encoder3_layer)
        encoder4 = BatchNormalization()(encoder4)
        encoder4 = Activation('relu')(encoder4)
        encoder4 = Conv2D(128, (3, 3), padding='same')(encoder4)
        encoder4 = BatchNormalization()(encoder4)
        encoder4 = Activation('relu')(encoder4)
        encoder4_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder4)
        # 32

        encoder5 = Conv2D(256, (3, 3), padding='same')(encoder4_layer)
        encoder5 = BatchNormalization()(encoder5)
        encoder5 = Activation('relu')(encoder5)
        encoder5 = Conv2D(256, (3, 3), padding='same')(encoder5)
        encoder5 = BatchNormalization()(encoder5)
        encoder5 = Activation('relu')(encoder5)
        encoder5_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder5)
        # 16

        encoder6 = Conv2D(512, (3, 3), padding='same')(encoder5_layer)
        encoder6 = BatchNormalization()(encoder6)
        encoder6 = Activation('relu')(encoder6)
        encoder6 = Conv2D(512, (3, 3), padding='same')(encoder6)
        encoder6 = BatchNormalization()(encoder6)
        encoder6 = Activation('relu')(encoder6)
        encoder6_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder6)
        # 8

        middle = Conv2D(1024, (3, 3), padding='same')(encoder6_layer)
        middle = BatchNormalization()(middle)
        middle = Activation('relu')(middle)
        middle = Conv2D(1024, (3, 3), padding='same')(middle)
        middle = BatchNormalization()(middle)
        middle = Activation('relu')(middle)
        # middle

        decoder6 = UpSampling2D((2, 2))(middle)
        decoder6 = concatenate([encoder6, decoder6], axis=3)
        decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
        decoder6 = BatchNormalization()(decoder6)
        decoder6 = Activation('relu')(decoder6)
        decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
        decoder6 = BatchNormalization()(decoder6)
        decoder6 = Activation('relu')(decoder6)
        decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
        decoder6 = BatchNormalization()(decoder6)
        decoder6 = Activation('relu')(decoder6)
        # 16

        decoder5 = UpSampling2D((2, 2))(decoder6)
        decoder5 = concatenate([encoder5, decoder5], axis=3)
        decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
        decoder5 = BatchNormalization()(decoder5)
        decoder5 = Activation('relu')(decoder5)
        decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
        decoder5 = BatchNormalization()(decoder5)
        decoder5 = Activation('relu')(decoder5)
        decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
        decoder5 = BatchNormalization()(decoder5)
        decoder5 = Activation('relu')(decoder5)
        # 32

        decoder4 = UpSampling2D((2, 2))(decoder5)
        decoder4 = concatenate([encoder4, decoder4], axis=3)
        decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
        decoder4 = BatchNormalization()(decoder4)
        decoder4 = Activation('relu')(decoder4)
        decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
        decoder4 = BatchNormalization()(decoder4)
        decoder4 = Activation('relu')(decoder4)
        decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
        decoder4 = BatchNormalization()(decoder4)
        decoder4 = Activation('relu')(decoder4)
        # 64

        decoder3 = UpSampling2D((2, 2))(decoder4)
        decoder3 = concatenate([encoder3, decoder3], axis=3)
        decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
        decoder3 = BatchNormalization()(decoder3)
        decoder3 = Activation('relu')(decoder3)
        decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
        decoder3 = BatchNormalization()(decoder3)
        decoder3 = Activation('relu')(decoder3)
        decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
        decoder3 = BatchNormalization()(decoder3)
        decoder3 = Activation('relu')(decoder3)
        # 128

        decoder2 = UpSampling2D((2, 2))(decoder3)
        decoder2 = concatenate([encoder2, decoder2], axis=3)
        decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
        decoder2 = BatchNormalization()(decoder2)
        decoder2 = Activation('relu')(decoder2)
        decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
        decoder2 = BatchNormalization()(decoder2)
        decoder2 = Activation('relu')(decoder2)
        decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
        decoder2 = BatchNormalization()(decoder2)
        decoder2 = Activation('relu')(decoder2)
        # 256

        decoder1 = UpSampling2D((2, 2))(decoder2)
        decoder1 = concatenate([encoder1, decoder1], axis=3)
        decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
        decoder1 = BatchNormalization()(decoder1)
        decoder1 = Activation('relu')(decoder1)
        decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
        decoder1 = BatchNormalization()(decoder1)
        decoder1 = Activation('relu')(decoder1)
        decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
        decoder1 = BatchNormalization()(decoder1)
        decoder1 = Activation('relu')(decoder1)
        # 512

        out = Conv2D(num_classes, (1, 1), activation='sigmoid')(decoder1)

        model = Model(inputs=[inputs], outputs=[out])

        model.compile(optimizer=RMSprop(learning_rate=0.0001),
                      loss=UNet.bce_dice_loss,
                      metrics=[UNet.dice_coeff])

        return model

    def train(self, X_train, y_train, X_val, y_val,
              patience_counter=2, batch_size=64, epochs=100):
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, \
            ReduceLROnPlateau
        super().train(X_train, y_train, X_val, y_val, patience_counter)

        checkpoint_file = os.path.join(self.workingdir, 'unet')
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

        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=(X_val, y_val),
                                 verbose=int(self.verbose))

        history_file = os.path.join(self.workingdir, 'history')
        history_file = Util.create_numbered_file(history_file, '.pickle')

        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        print('Model saved to', checkpoint_file)
        print('History saved to', history_file)

        return history

    def predict(self, X_test, y_test, threshold=0.5):
        predictions = self.model.predict(X_test)

        predictions[predictions >= threshold] = 1.0
        predictions[predictions < threshold] = 0.0

        scores = self.model.evaluate(X_test, y_test)

        return predictions, scores

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        import tensorflow as tf
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        return tf.keras.losses.binary_crossentropy(y_true_f, y_pred_f) + (
                1 - UNet.dice_coeff(y_true_f, y_pred_f))

    @staticmethod
    def dice_coeff(y_true, y_pred):
        import tensorflow as tf
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
        smoothing_const = 1e-9
        return (2. * intersection + smoothing_const) / (
                    tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(
                y_pred_f) + smoothing_const)