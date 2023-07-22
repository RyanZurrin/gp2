from .base_keras_segmentation_classifier import \
    BaseKerasSegmentationClassifier

from gp2.gp2.util import Util


class KVNet2D(BaseKerasSegmentationClassifier):
    """
    KVNet2D for binary segmentation
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 res_num_ini=1,
                 res_num_max=2,
                 activation='ReLU',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=False,
                 unpool=False,
                 name='kvnet2d',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        """
        vnet 2d

        vnet_2d(input_size, filter_num, n_labels,
                res_num_ini=1, res_num_max=3,
                activation='ReLU', output_activation='Softmax',
                batch_norm=False, pool=True, unpool=True, name='vnet')

        Milletari, F., Navab, N. and Ahmadi, S.A., 2016, October. V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. In 2016 fourth international conference
        on 3D vision (3DV) (pp. 565-571). IEEE.

        The Two-dimensional version is inspired by:
        https://github.com/FENGShuanglang/2D-Vnet-Keras

        Input
        ----------
            input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
            filter_num: a list that defines the number of filters for each \
                        down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                        The depth is expected as `len(filter_num)`.
            n_labels: number of output labels.
            res_num_ini: number of convolutional layers of the first first residual block (before downsampling).
            res_num_max: the max number of convolutional layers within a residual block.
            activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
            output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                               Default option is 'Softmax'.
                               if None is received, then linear activation is applied.
            batch_norm: True for batch normalization.
            pool: True or 'max' for MaxPooling2D.
                  'ave' for AveragePooling2D.
                  False for strided conv + batch norm + activation.
            unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                    'nearest' for Upsampling2D with nearest interpolation.
                    False for Conv2DTranspose + batch norm + activation.
            name: prefix of the created keras layers.

        Output
        ----------
            model: a keras model.

        * This is a modified version of V-net for 2-d inputw.
        * The original work supports `pool=False` only.
          If pool is True, 'max', or 'ave', an additional conv2d layer will be applied.
        * All the 5-by-5 convolutional kernels are changed (and fixed) to 3-by-3.
        """
        from keras import losses
        from tensorflow.keras import optimizers
        from keras_unet_collection import models
        super().__init__(verbose=verbose, workingdir=workingdir)

        self.input_size = input_size
        self.filter_num = filter_num or [32, 64, 128, 256]
        self.n_labels = n_labels
        self.res_num_ini = res_num_ini
        self.res_num_max = res_num_max
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.name = name
        self.optimizer = optimizer or optimizers.Adam(learning_rate=1e-4)
        self.loss = loss or losses.binary_crossentropy
        self.metric = metric or [Util.dice_coeff]

        self.model = models.vnet_2d(input_size=self.input_size,
                                    filter_num=self.filter_num,
                                    n_labels=self.n_labels,
                                    res_num_ini=self.res_num_ini,
                                    res_num_max=self.res_num_max,
                                    activation=self.activation,
                                    output_activation=self.output_activation,
                                    batch_norm=self.batch_norm,
                                    pool=self.pool,
                                    unpool=self.unpool,
                                    name=self.name)
        print('*** GP2  KVNet2D ***')
        print('Working directory:', self.workingdir)

        self.build()

        if self.verbose:
            print('Verbose mode active!')
            print(self)
            print('Model summary:')
            self.model.summary()
