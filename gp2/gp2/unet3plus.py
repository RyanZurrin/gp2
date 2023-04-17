from .classifier import Classifier
from .util import Util
import numpy as np
from glob import glob
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, UpSampling2D, GlobalMaxPooling2D, multiply
from tensorflow.keras.backend import max
from keras_unet_collection import models, base, utils


class Unet3Plus(Classifier):

    def __init__(self,
                 workingdir='/tmp',
                 verbose=False,
                 name='unet3plus',
                 activation='relu',

                 ):
        super().__init__(verbose=verbose, workingdir=workingdir)
