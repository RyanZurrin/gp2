from .util import Util
Util.disable_tensorflow_logging()

from gp2.gp2.classifiers.classifier import Classifier
from gp2.gp2.discriminators.cnndiscriminator import CNNDiscriminator
from .classifiers import *
from .discriminators import *
