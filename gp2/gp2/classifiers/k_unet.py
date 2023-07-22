from .base_keras_segmentation_classifier import \
    BaseKerasSegmentationClassifier
from gp2.gp2.util import Util


class KUNet(BaseKerasSegmentationClassifier):
    def __init__(self,
                 input_shape=(512, 512, 1),
                 num_classes=1,
                 activation='relu',
                 use_batch_norm=True,
                 upsample_mode='deconv',
                 dropout=0.3,
                 dropout_change_per_layer=0.0,
                 dropout_type='spatial',
                 use_dropout_on_upsampling=False,
                 use_attention=False,
                 filters=16,
                 num_layers=4,
                 output_activation='sigmoid',
                 name='kunet',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        """ Keras Custom U-Net model.

        custom_unet(input_shape, num_classes=1, activation='sigmoid', use_batch_norm=True,
        upsample_mode='deconv', dropout=0.5, dropout_change_per_layer=0.0,
        dropout_type='spatial', use_dropout_on_upsampling=False, use_attention=False,
        filters=16, num_layers=4, output_activation='sigmoid')


        Parameters
        ----------
        input_shape : tuple
            3D Tensor of shape (x, y, num_channels)
        num_classes : int
            Unique classes in the output mask. Should be set to 1 for binary segmentation
        activation : str
            A keras.activations.Activation to use. ReLu by default.
        use_batch_norm : bool
             Whether to use Batch Normalisation across the channel axis between
             convolutional layers
        upsample_mode : str
            (one of "deconv" or "simple"): Whether to use transposed convolutions
            or simple upsampling in the decoder part
        dropout : (float between 0. and 1.)
            Amount of dropout after the initial convolutional
            block. Set to 0. to turn Dropout off
        dropout_change_per_layer : (float between 0. and 1.)
            Factor to add to the Dropout after each convolutional block
        dropout_type : (one of "spatial" or "standard")
            Type of Dropout to apply. Spatial is recommended for CNNs [2]
        use_dropout_on_upsampling : bool
            Whether to use dropout in the decoder part of the network
        use_attention : bool
             Whether to use an attention dynamic when concatenating with the
             skip-connection, implemented as proposed by Oktay et al. [3]
        filters : int
            Convolutional filters in the initial convolutional block. Will be doubled every block
        num_layers : int
            Number of total layers in the encoder not including the bottleneck layer
        output_activation : str
            A keras.activations.Activation to use. Sigmoid by default for binary segmentation

        Returns
        -------
        model : (keras.models.Model)
            The built U-Net

        Raises
        ------
        ValueError
            If dropout_type is not one of "spatial" or "standard"

        References
        ----------
        [1]: https://arxiv.org/abs/1505.04597
        [2]: https://arxiv.org/pdf/1411.4280.pdf
        [3]: https://arxiv.org/abs/1804.03999
        """
        from keras import losses
        from tensorflow.keras import optimizers
        from keras_unet.models import custom_unet as custom

        super().__init__(verbose=verbose, workingdir=workingdir)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.upsample_mode = upsample_mode
        self.dropout = dropout
        self.dropout_change_per_layer = dropout_change_per_layer
        self.dropout_type = dropout_type
        self.use_dropout_on_upsampling = use_dropout_on_upsampling
        self.use_attention = use_attention
        self.filters = filters
        self.num_layers = num_layers
        self.output_activation = output_activation
        self.name = name
        self.optimizer = optimizer or optimizers.Adam(learning_rate=1e-4)
        self.loss = loss or losses.binary_crossentropy
        self.metric = metric or [Util.dice_coeff]
        self.model = custom(input_shape=self.input_shape,
                            num_classes=self.num_classes,
                            activation=self.activation,
                            use_batch_norm=self.use_batch_norm,
                            upsample_mode=self.upsample_mode,
                            dropout=self.dropout,
                            dropout_change_per_layer=self.dropout_change_per_layer,
                            dropout_type=self.dropout_type,
                            use_dropout_on_upsampling=self.use_dropout_on_upsampling,
                            use_attention=self.use_attention,
                            filters=self.filters,
                            num_layers=self.num_layers,
                            output_activation=self.output_activation
                            )

        print('*** GP2 Keras UNet ***')
        print('Working directory:', self.workingdir)

        self.build()

        if verbose:
            print('Verbose mode active!')
            print(self)
            print('Model summary:')
            self.model.summary()
