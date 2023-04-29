from keras import backend as K
from sklearn.metrics import roc_auc_score
from utils import preprocess_for_auc
from keras.layers import Layer, InputSpec
import pdb

def keras_im_shape(h, w, n_channels, data_format=K.image_data_format()):
    if data_format == 'channels_first':
        return (n_channels, h, w)
    else:
        return (h, w, n_channels)

def get_channels_axis():
    return -1 if K.image_data_format() == 'channels_last' else -3

def neg_auc(y, y_hat, axis=-1):
    y, y_hat = preprocess_for_auc(y, y_hat, axis)
    return -1 * roc_auc_score(y, y_hat)

def stepwise_lr(epoch, init_lr, lr_epochs):
    lr = init_lr
    for tup in lr_epochs:
        if epoch >= tup[0]:
            lr = tup[1]
    return lr

class BiasLayer(Layer):
    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=input_shape[1:],
                                      initializer='zeros',
                                      name='bias')
        self.input_spec = InputSpec(ndim=4, axes={1: input_shape[1]})
        self.built = True

    def call(self, inputs):
        return K.add(inputs, self.bias)

class BiasGradLayer(Layer):
    def __init__(self, output_shape, **kwargs):
        super(BiasGradLayer, self).__init__(**kwargs)
        self.output_shape = output_shape

    def call(self, inputs):
        pdb.set_trace()
        return K.add(inputs, self.bias)

    def compute_output_shape(self, input_shape):
        return self.output_shape

def gray2rgb(x):
    #axis = -3 if K.image_data_format() == 'channels_first' else -1 # tf throws error with negative axis
    axis = 1 if K.image_data_format() == 'channels_first' else 3
    return K.repeat_elements(x, 3, axis)

def gray2rgb_shape(input_shape):
    out_shape = list(input_shape)
    axis = -3 if K.image_data_format() == 'channels_first' else -1
    out_shape[axis] = 3
    return tuple(out_shape)

def slice_layer(x):
    return x[:, :, -1:]

def slice_layer_shape(input_shape):
    in_shape = list(input_shape)
    in_shape[-1] = 1
    return tuple(in_shape)
