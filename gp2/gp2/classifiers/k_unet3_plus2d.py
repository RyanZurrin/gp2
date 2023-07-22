from .base_keras_segmentation_classifier import \
    BaseKerasSegmentationClassifier
from gp2.gp2.util import Util


class KUNet3Plus2D(BaseKerasSegmentationClassifier):
    """
    Keras implementation of UNET 3+ with an optional ImageNet-trained backbone.
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 n_labels=1,
                 filter_num_down=None,
                 filter_num_skip=None,
                 filter_num_aggregate=None,
                 stack_num_down=1,
                 stack_num_up=3,
                 activation='ReLU',
                 output_activation='Sigmoid',
                 batch_norm=False,
                 pool=True,
                 unpool=True,
                 deep_supervision=False,
                 name='kunet3plus2d',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        """
        UNET 3+ with an optional ImageNet-trained backbone.

        unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto',
                      stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                      batch_norm=False, pool=True, unpool=True, deep_supervision=False,
                      backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')

        ----------
        Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020.
        UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.
        In ICASSP 2020-2020 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.

        Input
        ----------
            input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
            filter_num_down: a list that defines the number of filters for each
                             downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                             the network depth is expected as `len(filter_num_down)`
            filter_num_skip: a list that defines the number of filters after each
                             full-scale skip connection. Number of elements is expected to be `depth-1`.
                             i.e., the bottom level is not included.
                             * Huang et al. (2020) applied the same numbers for all levels.
                               e.g., `[64, 64, 64, 64]`.
            filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
            stack_num_down: number of convolutional layers per downsampling level/block.
            stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.
            activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'
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
            deep_supervision: True for a model that supports deep supervision. Details see Huang et al. (2020).
            name: prefix of the created keras model and its layers.

            ---------- (keywords of backbone options) ----------
            backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                           None (default) means no backbone.
                           Currently supported backbones are:
                           (1) VGG16, VGG19
                           (2) ResNet50, ResNet101, ResNet152
                           (3) ResNet50V2, ResNet101V2, ResNet152V2
                           (4) DenseNet121, DenseNet169, DenseNet201
                           (5) EfficientNetB[0-7]
            weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                     or the path to the weights file to be loaded.
            freeze_backbone: True for a frozen backbone.
            freeze_batch_norm: False for not freezing batch normalization layers.

        * The Classification-guided Module (CGM) is not implemented.
          See https://github.com/yingkaisha/keras-unet-collection/tree/main/examples for a relevant example.
        * Automated mode is applied for determining `filter_num_skip`, `filter_num_aggregate`.
        * The default output activation is sigmoid, consistent with Huang et al. (2020).
        * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
        * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.

        Output
        ----------
            model: a keras model.

        """
        from keras import losses
        from tensorflow.keras import optimizers
        from keras_unet_collection import models

        super().__init__(verbose=verbose, workingdir=workingdir)

        print(f'KUNet3Plus2D: {optimizer}, {loss}, {metric}')

        self.input_size = input_size
        self.n_labels = n_labels
        self.filter_num_down = filter_num_down or [32, 64, 128, 256]
        self.filter_num_skip = filter_num_skip or [16, 16, 16, 16]
        self.filter_num_aggregate = filter_num_aggregate or 64
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.deep_supervision = deep_supervision
        self.name = name
        self.optimizer = optimizer or optimizers.Adam(learning_rate=1e-4)
        self.loss = loss or losses.binary_crossentropy
        self.metric = metric or [Util.dice_coeff]

        self.model = models.unet_3plus_2d(input_size=self.input_size,
                                          n_labels=self.n_labels,
                                          filter_num_down=self.filter_num_down,
                                          filter_num_skip=self.filter_num_skip,
                                          filter_num_aggregate=self.filter_num_aggregate,
                                          stack_num_down=self.stack_num_down,
                                          stack_num_up=self.stack_num_up,
                                          activation=self.activation,
                                          output_activation=self.output_activation,
                                          batch_norm=self.batch_norm,
                                          pool=self.pool,
                                          unpool=self.unpool,
                                          deep_supervision=self.deep_supervision,
                                          backbone=None,
                                          weights=None,
                                          freeze_backbone=True,
                                          freeze_batch_norm=True,
                                          name=self.name)
        print('*** GP2  KUNet3Plus2D ***')
        print('Working directory:', self.workingdir)

        self.build()

        if self.verbose:
            print('Verbose mode active!')
            print(self)
            print('Model summary:')
            print(self.model.summary())
