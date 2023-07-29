from gp2.utils import Util

Util.disable_tensorflow_logging()

from .discriminator import Discriminator
from .base_cnn_discriminator import BaseCNNDiscriminator
from .cnndiscriminator import CNNDiscriminator
from .cnn_discriminator_plus import CNNDiscriminatorPLUS
