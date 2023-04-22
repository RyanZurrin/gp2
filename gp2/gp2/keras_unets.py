import os
import pickle

from keras import losses, metrics
from tensorflow.keras import callbacks, optimizers

from .base_keras_segmentation_classifier import BaseKerasSegmentationClassifier
from .util import Util
from keras_unet_collection import models, base, utils
import tensorflow as tf

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


class KATTUnet2D(BaseKerasSegmentationClassifier):
    """
    Attention U-net with an optional ImageNet backbone
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 stack_num_down=3,
                 stack_num_up=3,
                 activation='ReLU',
                 atten_activation='ReLU',
                 attention='add',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool='ave',
                 unpool='nearest',
                 backbone=None,
                 weights='imagenet',
                 freeze_backbone=True,
                 freeze_batch_norm=True,
                 name='attunet',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        '''
        Attention U-net with an optional ImageNet backbone

        att_unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2, activation='ReLU',
                    atten_activation='ReLU', attention='add', output_activation='Softmax', batch_norm=False, pool=True, unpool=True,
                    backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='att-unet')

        ----------
        Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B.
        and Glocker, B., 2018. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.

        Input
        ----------
            input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
            filter_num: a list that defines the number of filters for each \
                        down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                        The depth is expected as `len(filter_num)`.
            n_labels: number of output labels.
            stack_num_down: number of convolutional layers per downsampling level/block.
            stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
            activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
            output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                               Default option is 'Softmax'.
                               if None is received, then linear activation is applied.
            atten_activation: a nonlinear atteNtion activation.
                        The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
            attention: 'add' for additive attention. 'multiply' for multiplicative attention.
                       Oktay et al. 2018 applied additive attention.
            batch_norm: True for batch normalization.
            pool: True or 'max' for MaxPooling2D.
                  'ave' for AveragePooling2D.
                  False for strided conv + batch norm + activation.
            unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                    'nearest' for Upsampling2D with nearest interpolation.
                    False for Conv2DTranspose + batch norm + activation.
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

        Output
        ----------
            model: a keras model

        '''
        super().__init__(verbose=verbose, workingdir=workingdir)

        if filter_num is None:
            filter_num = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        if optimizer is None:
            self.optimizer = optimizers.Adam(learning_rate=1e-5)

        if loss is None:
            self.loss = losses.binary_crossentropy

        if metric is None:
            self.metric = [dice_coef]

        self.input_size = input_size
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.atten_activation = atten_activation
        self.attention = attention
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.backbone = backbone
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        self.name = name
        self.model = models.att_unet_2d(self.input_size,
                                        filter_num=self.filter_num,
                                        n_labels=self.n_labels,
                                        stack_num_down=self.stack_num_down,
                                        stack_num_up=self.stack_num_up,
                                        activation=self.activation,
                                        atten_activation=self.atten_activation,
                                        attention=self.attention,
                                        output_activation=self.output_activation,
                                        batch_norm=self.batch_norm,
                                        pool=self.pool, unpool=self.unpool,
                                        backbone=self.backbone,
                                        weights=self.weights,
                                        freeze_backbone=self.freeze_backbone,
                                        freeze_batch_norm=self.freeze_batch_norm,
                                        name=self.name)
        print('*** GP2 KATTUnet2D ***')
        print('Working directory:', self.workingdir)

        if verbose:
            print('Verbose mode active!')

        self.build()

        if self.verbose:
            self.model.summary()


class KUNet2D(BaseKerasSegmentationClassifier):
    """
    KU-Net 2D model.
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 stack_num_down=2,
                 stack_num_up=2,
                 activation='ReLU',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=True,
                 unpool=True,
                 backbone=None,
                 weights='imagenet',
                 freeze_backbone=True,
                 freeze_batch_norm=True,
                 name='unet',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        """
        U-net with an optional ImageNet-trained bakcbone.

        unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True,
                backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet')

        ----------
        Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation.
        In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

        Input
        ----------
            input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
            filter_num: a list that defines the number of filters for each \
                        down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                        The depth is expected as `len(filter_num)`.
            n_labels: number of output labels.
            stack_num_down: number of convolutional layers per downsampling level/block.
            stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
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

        Output
        ----------
            model: a keras model.

        """
        super().__init__(verbose=verbose, workingdir=workingdir)

        if filter_num is None:
            filter_num = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        if optimizer is None:
            optimizer = optimizers.Adam(learning_rate=1e-3)
        if loss is None:
            loss = losses.binary_crossentropy

        if metric is None:
            metric = [dice_coef]

        self.input_size = input_size
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.backbone = backbone
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.model = models.unet_2d(self.input_size,
                                    filter_num=self.filter_num,
                                    n_labels=self.n_labels,
                                    stack_num_down=self.stack_num_down,
                                    stack_num_up=self.stack_num_up,
                                    activation=self.activation,
                                    output_activation=self.output_activation,
                                    batch_norm=self.batch_norm,
                                    pool=self.pool, unpool=self.unpool,
                                    backbone=self.backbone,
                                    weights=self.weights,
                                    freeze_backbone=self.freeze_backbone,
                                    freeze_batch_norm=self.freeze_batch_norm,
                                    name=self.name)

        print('*** GP2  KUNet2D ***')
        print('Working directory:', self.workingdir)

        if verbose:
            print('Verbose mode active!')

        self.build()

        if verbose:
            self.model.summary()


class KUNetPlus2D(BaseKerasSegmentationClassifier):
    """
    Keras U-net++ 2D model.
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 stack_num_down=2,
                 stack_num_up=2,
                 activation='ReLU',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=False,
                 unpool=False,
                 backbone=None,
                 weights=None,
                 freeze_backbone=True,
                 freeze_batch_norm=True,
                 name='unet',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        """
        U-net++ with an optional ImageNet-trained backbone.

        unet_plus_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                     activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, deep_supervision=False,
                     backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='xnet')

        ----------
        Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N. and Liang, J., 2018. Unet++: A nested u-net architecture
        for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning
        for Clinical Decision Support (pp. 3-11). Springer, Cham.

        Input
        ----------
            input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
            filter_num: a list that defines the number of filters for each \
                        down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                        The depth is expected as `len(filter_num)`.
            n_labels: number of output labels.
            stack_num_down: number of convolutional layers per downsampling level/block.
            stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
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
            deep_supervision: True for a model that supports deep supervision. Details see Zhou et al. (2018).
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

        Output
        ----------
            model: a keras model.

        """
        super().__init__(verbose=verbose, workingdir=workingdir)

        if filter_num is None:
            filter_num = [32, 64, 128, 256, 512]

        if optimizer is None:
            optimizer = optimizers.Adam(learning_rate=1e-4)

        if loss is None:
            loss = losses.binary_crossentropy

        if metric is None:
            metric = [dice_coef]

        self.input_size = input_size
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.backbone = backbone
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.model = models.unet_plus_2d(self.input_size,
                                         filter_num=self.filter_num,
                                         n_labels=self.n_labels,
                                         stack_num_down=self.stack_num_down,
                                         stack_num_up=self.stack_num_up,
                                         activation=self.activation,
                                         output_activation=self.output_activation,
                                         batch_norm=self.batch_norm,
                                         pool=self.pool, unpool=self.unpool,
                                         backbone=self.backbone,
                                         weights=self.weights,
                                         freeze_backbone=self.freeze_backbone,
                                         freeze_batch_norm=self.freeze_batch_norm,
                                         name=self.name)
        print('*** GP2  KUNetPlus2D ***')
        print('Working directory:', self.workingdir)

        if verbose:
            print('Verbose mode active!')

        self.build()

        if verbose:
            self.model.summary()


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
                 stack_num_down=2,
                 stack_num_up=2,
                 activation='ReLU',
                 output_activation='Softmax',
                 batch_norm=True,
                 pool=True,
                 unpool=True,
                 deep_supervision=False,
                 name='unet3plus',
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
        super().__init__(verbose=verbose, workingdir=workingdir)

        if filter_num_down is None:
            filter_num_down = [16, 32, 64, 128, 256]
        if filter_num_skip is None:
            filter_num_skip = 'auto'
        if filter_num_aggregate is None:
            filter_num_aggregate = 'auto'
        if optimizer is None:
            optimizer = optimizers.Adam()
        if loss is None:
            loss = losses.binary_crossentropy
        if metric is None:
            metric = [dice_coef]

        print(f'KUNet3Plus2D: {optimizer}, {loss}, {metric}')

        self.input_size = input_size
        self.n_labels = n_labels
        self.filter_num_down = filter_num_down
        self.filter_num_skip = filter_num_skip
        self.filter_num_aggregate = filter_num_aggregate
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.deep_supervision = deep_supervision
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric

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

        if verbose:
            print('Verbose mode active!')
        self.build()

        if self.verbose:
            print(self.model.summary())


class KVNet2D(BaseKerasSegmentationClassifier):
    """
    KVNet2D for binary segmentation
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 res_num_ini=1,
                 res_num_max=3,
                 activation='PReLU',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=False,
                 unpool=False,
                 name='vnet',
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
        super().__init__(verbose=verbose, workingdir=workingdir)

        if filter_num is None:
            filter_num = [16, 32, 64, 128, 256, 512, 1024]

        if optimizer is None:
            optimizer = optimizers.Adam(learning_rate=1e-4)

        if loss is None:
            loss = losses.binary_crossentropy

        if metric is None:
            metric = [dice_coef]

        self.input_size = input_size
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.res_num_ini = res_num_ini
        self.res_num_max = res_num_max
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric

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

        if verbose:
            print('Verbose mode active!')
        self.build()

        if self.verbose:
            self.model.summary()


class KResUNet2D(BaseKerasSegmentationClassifier):
    """
    KResUNet2D
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 dilation_num=None,
                 n_labels=1,
                 aspp_num_down=32,
                 aspp_num_up=16,
                 activation='ReLU',
                 output_activation='Softmax',
                 batch_norm=True,
                 pool=True,
                 unpool=True,
                 name='resunet',
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
        super().__init__(verbose=verbose, workingdir=workingdir)

        if filter_num is None:
            filter_num = [16, 32, 64, 128, 256]

            if dilation_num is None:
                dilation_num = [1, 3, 15, 31]

        if optimizer is None:
            optimizer = optimizers.Adam(learning_rate=1e-4)

        if loss is None:
            loss = losses.binary_crossentropy

        if metric is None:
            metric = [metrics.binary_accuracy]

        self.input_size = input_size
        self.filter_num = filter_num
        self.dilation_num = dilation_num
        self.n_labels = n_labels
        self.aspp_num_down = aspp_num_down
        self.aspp_num_up = aspp_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
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

        if verbose:
            print('Verbose mode active!')
        self.build()

        if verbose:
            self.model.summary()


def dice_coef(y_true, y_pred, smooth=1e-9):
    """ Calculate the dice coefficient.
    Parameters
    ----------
    y_true : numpy.ndarray
        The true masks.
    y_pred : numpy.ndarray
        The predicted masks.
    smooth : float
        The smoothing factor.

    Returns
    -------
    float
        The dice coefficient.
    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def bce_dice_loss(y_true, y_pred):
    """ Calculate the loss.
    Parameters
    ----------
    y_true : numpy.ndarray
        The true masks.
    y_pred : numpy.ndarray
        The predicted masks.

    Returns
    -------
    float
        The loss.
    """
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + \
        (1 - dice_coef(y_true, y_pred))
