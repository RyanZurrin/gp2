from .base_cnn_discriminator import BaseCNNDiscriminator, CustomMaskLayer


class CNNDiscriminatorPLUS(BaseCNNDiscriminator):
    def __init__(self,
                 image_shape=(512, 512, 1),
                 num_classes=2,
                 mask_layer=None,
                 conv_layers=None,
                 dense_layers=None,
                 optimizer=None,
                 loss_function=None,
                 metrics=None,
                 activation='leaky_relu',
                 alpha=0.1,
                 kernel_regularizer=None,
                 workingdir='/tmp',
                 verbose=True):
        """ Initializes the class.

        :param image_shape: The shape of the input images.
        :param num_classes: The number of classes.
        :param mask_layer: The mask layer.
        :param conv_layers: The convolution layers.
        :param dense_layers: The dense layers.
        :param optimizer: The optimizer.
        :param loss_function: The loss function.
        :param metrics: The metrics.
        :param activation: The activation function.
        :param alpha: The alpha value.
        :param kernel_regularizer: The kernel regularizer.
        :param workingdir: The working directory.
        :param verbose: The verbosity.
        """
        from tensorflow.keras import losses
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l1_l2
        super().__init__(workingdir)
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.mask_layer = mask_layer
        self.conv_layers = conv_layers or [
            (16, (3, 3), 0.25),
            (32, (3, 3), 0.25),
            (64, (3, 3), 0.25),
            (128, (3, 3), 0.4),
            (256, (3, 3), 0.4)
        ]
        self.dense_layers = dense_layers or [(512, 0.5)]
        self.optimizer = optimizer or Adam()
        self.loss_function = loss_function or losses.CategoricalCrossentropy()
        self.metrics = metrics or ['accuracy']
        self.activation = activation
        self.alpha = alpha
        self.kernel_regularizer = kernel_regularizer or l1_l2(l1=0.0, l2=0.0)
        self.workingdir = workingdir
        self.verbose = verbose

        self.model = self.build()

    def create_convolution_layers(self, input_img, input_shape=None):
        """ Creates the convolution layers.

        :param input_img: The input image.
        :param input_shape: The shape of the input image.
        :return: The convolution layers.
        """
        from tensorflow.keras import layers
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dropout
        model = input_img
        for filters, kernel_size, dropout_rate in self.conv_layers:
            model = Conv2D(filters, kernel_size, padding='same',
                           kernel_regularizer=self.kernel_regularizer)(model)
            if self.activation == 'leaky_relu':
                model = LeakyReLU(alpha=self.alpha)(model)
            else:
                model = layers.Activation(self.activation)(model)
            model = MaxPooling2D((2, 2), padding='same')(model)
            model = Dropout(dropout_rate)(model)

        return model

    def build(self):
        """ Builds the model.

        :return: The model.
        """
        from tensorflow.keras import layers
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, concatenate, LeakyReLU, \
            Dropout, Flatten, Dense
        image_input = Input(shape=self.image_shape)
        image_model = self.create_convolution_layers(image_input)

        mask_input = Input(shape=self.image_shape)
        # TODO: May need to remove the CustomMaskLayer, just testing
        mask_model = CustomMaskLayer()(mask_input)
        mask_model = self.create_convolution_layers(mask_model)

        conv = concatenate([image_model, mask_model])

        conv = Flatten()(conv)

        for units, dropout_rate in self.dense_layers:
            conv = Dense(units, kernel_regularizer=self.kernel_regularizer)(
                conv)
            if self.activation == 'leaky_relu':
                conv = LeakyReLU(alpha=self.alpha)(conv)
            else:
                conv = layers.Activation(self.activation)(conv)
            conv = Dropout(dropout_rate)(conv)

        output = Dense(self.num_classes, activation='softmax')(conv)

        model = Model(inputs=[image_input, mask_input], outputs=[output])

        model.compile(optimizer=self.optimizer,
                      loss=self.loss_function,
                      metrics=self.metrics)

        return model
