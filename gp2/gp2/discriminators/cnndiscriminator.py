from .base_cnn_discriminator import BaseCNNDiscriminator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, \
    LeakyReLU, Dropout, Flatten, Dense


class CNNDiscriminator(BaseCNNDiscriminator):

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