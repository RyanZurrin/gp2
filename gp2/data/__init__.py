import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
from .point import Point
from .collection import Collection
from .manager import Manager
