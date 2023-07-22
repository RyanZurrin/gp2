from gp2.gp2.util import Util
from .base_keras_segmentation_classifier import \
    BaseKerasSegmentationClassifier



class KATTUnet2D(BaseKerasSegmentationClassifier):
    """
    Attention U-net with an optional ImageNet backbone
    """

    def __init__(self,
                 input_size=(512, 512, 1),
                 filter_num=None,
                 n_labels=1,
                 stack_num_down=2,
                 stack_num_up=2,
                 activation='ReLU',
                 atten_activation='ReLU',
                 attention='add',
                 output_activation='Sigmoid',
                 batch_norm=True,
                 pool=True,
                 unpool=False,
                 backbone=None,
                 weights='imagenet',
                 freeze_backbone=True,
                 freeze_batch_norm=True,
                 name='kattunet2d',
                 optimizer=None,
                 loss=None,
                 metric=None,
                 verbose=False,
                 workingdir='/tmp',
                 ):
        """
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

        """
        from keras import losses
        from tensorflow.keras import optimizers
        from keras_unet_collection import models
        super().__init__(verbose=verbose, workingdir=workingdir)

        self.input_size = input_size
        self.filter_num = filter_num or [16, 32, 64, 128, 256]
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
        self.optimizer = optimizer or optimizers.Adam(learning_rate=1e-4)
        self.loss = loss or losses.binary_crossentropy
        self.metric = metric or [Util.dice_coeff]
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
        print('*** GP2 KATTUNet2D ***')
        print('Working directory:', self.workingdir)

        self.build()

        if self.verbose:
            print('Verbose mode active!')
            print(self)
            print('Model summary:')
            self.model.summary()
