from .base_keras_segmentation_classifier import \
    BaseKerasSegmentationClassifier
from gp2.gp2.util import Util


class KR2UNet2dD(BaseKerasSegmentationClassifier):
    """
    Keras UNet2D
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 stack_num_down=2,
                 stack_num_up=2,
                 recur_num=2,
                 activation='ReLU',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=True,
                 unpool=False,
                 name='kr2unet2d',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        """
        Recurrent Residual (R2) U-Net

        r2_unet_2d(input_size, filter_num, n_labels,
                   stack_num_down=2, stack_num_up=2, recur_num=2,
                   activation='ReLU', output_activation='Softmax',
                   batch_norm=False, pool=True, unpool=True, name='r2_unet')

        ----------
        Alom, M.Z., Hasan, M., Yakopcic, C., Taha, T.M. and Asari, V.K., 2018. Recurrent residual convolutional neural network
        based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955.

        Input
        ----------
            input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
            filter_num: a list that defines the number of filters for each \
                        down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                        The depth is expected as `len(filter_num)`.
            n_labels: number of output labels.
            stack_num_down: number of stacked recurrent convolutional layers per downsampling level/block.
            stack_num_down: number of stacked recurrent convolutional layers per upsampling level/block.
            recur_num: number of recurrent iterations.
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

        """
        from keras import losses
        from tensorflow.keras import optimizers
        from keras_unet_collection import models
        super().__init__(verbose=verbose, workingdir=workingdir)

        self.input_size = input_size
        self.filter_num = filter_num or [64, 128, 256, 512]
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.recur_num = recur_num
        self.n_labels = n_labels
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.name = name
        self.optimizer = optimizer or optimizers.Adam(learning_rate=1e-4)
        self.loss = loss or losses.binary_crossentropy
        self.metric = metric or [Util.dice_coeff]
        self.model = models.r2_unet_2d(input_size=self.input_size,
                                       filter_num=self.filter_num,
                                       n_labels=self.n_labels,
                                       stack_num_down=self.stack_num_down,
                                       stack_num_up=self.stack_num_up,
                                       recur_num=self.recur_num,
                                       activation=self.activation,
                                       output_activation=self.output_activation,
                                       batch_norm=self.batch_norm,
                                       pool=self.pool,
                                       unpool=self.unpool,
                                       name=self.name)

        print('*** GP2 KR2UNet2dD ***')
        print('Working directory:', self.workingdir)

        self.build()

        if verbose:
            print('Verbose mode active!')
            print(self)
            print('Model summary:')
            self.model.summary()
