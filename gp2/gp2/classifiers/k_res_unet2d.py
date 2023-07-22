from .base_keras_segmentation_classifier import \
    BaseKerasSegmentationClassifier
from gp2.gp2.util import Util


class KResUNet2D(BaseKerasSegmentationClassifier):
    """
    KResUNet2D
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 dilation_num=None,
                 n_labels=1,
                 aspp_num_down=None,
                 aspp_num_up=None,
                 activation='ReLU',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=True,
                 unpool=False,
                 name='kresunet2d',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        """
        ResUNet-a

        resunet_a_2d(input_size, filter_num, dilation_num, n_labels,
                     aspp_num_down=256, aspp_num_up=128, activation='ReLU', output_activation='Softmax',
                     batch_norm=True, pool=True, unpool=True, name='resunet')

        ----------
        Diakogiannis, F.I., Waldner, F., Caccetta, P. and Wu, C., 2020. Resunet-a: a deep learning framework for
        semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162, pp.94-114.

        Input
        ----------
            input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
            filter_num: a list that defines the number of filters for each \
                        down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                        The depth is expected as `len(filter_num)`.
            dilation_num: an iterable that defines the dilation rates of convolutional layers.
                          Diakogiannis et al. (2020) suggested `[1, 3, 15, 31]`.
                          * `dilation_num` can be provided as 2d iterables, with the second dimension matches
                          the model depth. e.g., for len(filter_num) = 4; dilation_num can be provided as:
                          `[[1, 3, 15, 31], [1, 3, 15], [1,], [1,]]`.
                          * If `dilation_num` is not provided per down-/upsampling level, then the automated
                          determinations will be applied.
            n_labels: number of output labels.
            aspp_num_down: number of Atrous Spatial Pyramid Pooling (ASPP) layer filters after the last downsampling block.
            aspp_num_up: number of ASPP layer filters after the last upsampling block.
            activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
            output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                               Default option is 'Softmax'.
                               if None is received, then linear activation is applied.
            batch_norm: True for batch normalization.
            unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                    'nearest' for Upsampling2D with nearest interpolation.
                    False for Conv2DTranspose + batch norm + activation.
            name: prefix of the created keras layers.

        Output
        ----------
            model: a keras model.

        * Downsampling is achieved through strided convolutional layers with 1-by-1 kernels in Diakogiannis et al., (2020),
          and is here is achieved either with pooling layers or strided convolutional layers with 2-by-2 kernels.
        * `resunet_a_2d` does not support NoneType input shape.

        """
        from keras import losses
        from tensorflow.keras import optimizers
        from keras_unet_collection import models
        super().__init__(verbose=verbose, workingdir=workingdir)

        self.input_size = input_size
        self.filter_num = filter_num or [16, 32, 64, 128]
        self.dilation_num = dilation_num or [[1, 3, 15, 31], [1, 3, 15], [1, ],
                                             [1, ]]
        self.n_labels = n_labels
        self.aspp_num_down = aspp_num_down
        self.aspp_num_up = aspp_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.name = name
        self.optimizer = optimizer or optimizers.Adam(learning_rate=1e-4)
        self.loss = loss or losses.binary_crossentropy
        self.metric = metric or [Util.dice_coeff]
        self.model = models.resunet_a_2d(input_size=self.input_size,
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
        print('*** GP2  KResUNet2D ***')
        print('Working directory:', self.workingdir)

        self.build()

        if verbose:
            print('Verbose mode active!')
            print(self)
            print('Model summary:')
            self.model.summary()
