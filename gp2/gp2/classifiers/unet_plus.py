
from tensorflow.keras import layers, models, optimizers, losses
from .base_keras_segmentation_classifier import BaseKerasSegmentationClassifier
from gp2.gp2.util import Util


class UNetPLUS(BaseKerasSegmentationClassifier):

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
                 metrics=None,
                 name='unetplus',
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
            name : str
                The name of the model.
            verbose : bool
                Whether to print the model summary.
            workingdir : str
                The working directory to use for saving the model.

            """
        super().__init__(verbose=verbose, workingdir=workingdir)

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
        self.name = name
        self.optimizer = optimizer or optimizers.Adam(learning_rate=1e-4)
        self.loss = loss or losses.binary_crossentropy
        self.metrics = metrics or [Util.dice_coeff]

        print('*** GP2  UNetPLUS ***')
        print('Working directory:', self.workingdir)

        self.model = self.build()

        if verbose:
            print('Verbose mode active!')
            print(vars(self))
            self.model.summary()

    def build(self):
        """ Build the model.

        Returns
        -------
        model : keras.Model
            The model.
        """
        inputs = layers.Input(shape=self.input_shape)

        # Contracting path (encoder)
        x = inputs
        skips = []
        for i in range(self.depth):
            for _ in range(self.num_layers):
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
        for _ in range(self.num_layers):
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
            for _ in range(self.num_layers):
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

        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)

        return model

