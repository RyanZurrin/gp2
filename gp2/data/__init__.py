import warnings
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# import logging
# import tensorflow as tf
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)
# from tensorflow.python.keras.utils.generic_utils import CustomMaskWarning
# warnings.filterwarnings('ignore', category=CustomMaskWarning)
from .point import Point
from .collection import Collection
from .manager import Manager
