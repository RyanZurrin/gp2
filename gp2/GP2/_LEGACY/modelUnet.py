import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
import split
import numpy as np


def classifierDataset(noisedImages, labels):
    
    noisedImages = np.asarray(noisedImages, dtype=np.float32)  
    
    n_vals, perm = split.dataSplit(len(noisedImages))
    
    trainA_x, trainA_y = noisedImages[perm[0:n_vals[0]], ...], labels[perm[0:n_vals[0]], ...]
    valA_x, valA_y = noisedImages[perm[n_vals[0]:n_vals[1]], ...], labels[perm[n_vals[0]:n_vals[1]], ...]
    testA_x, testA_y = noisedImages[perm[n_vals[1]:n_vals[2]], ...], labels[perm[n_vals[1]:n_vals[2]], ...]
    human_x, human_y = noisedImages[perm[n_vals[2]:n_vals[3]], ...], labels[perm[n_vals[2]:n_vals[3]], ...]
    
    return (trainA_x, trainA_y), (valA_x, valA_y), (testA_x, testA_y), (human_x, human_y)



#defining function for calculation of metric dice coefficient
def dice_coeff(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    smoothing_const = 1e-9
    return (2. * intersection + smoothing_const) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + smoothing_const)

#defining function for calculation of loss function: binary cross entropy + dice loss
def bce_dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1-dice_coeff(y_true, y_pred))


# def dice_coeff(y_true, y_pred):
#     smooth = 1.
#     y_pred = tf.where(y_pred > 0.5, 1.0, 0.0)
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return score


# def dice_loss(y_true, y_pred):
#     loss = 1 - dice_coeff(y_true, y_pred)
#     return loss


# def bce_dice_loss(y_true, y_pred):
#     loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#     return loss

def build_unet_model(input_shape=(512, 512, 1),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 512
    
    encoder1 = Conv2D(16, (3, 3), padding='same', input_shape=input_shape)(inputs)
    encoder1 = BatchNormalization()(encoder1)
    encoder1 = Activation('relu')(encoder1)
    encoder1 = Conv2D(16, (3, 3), padding='same')(encoder1)
    encoder1 = BatchNormalization()(encoder1)
    encoder1 = Activation('relu')(encoder1)
    encoder1_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder1)
    # 256

    encoder2 = Conv2D(32, (3, 3), padding='same')(encoder1_layer)
    encoder2 = BatchNormalization()(encoder2)
    encoder2 = Activation('relu')(encoder2)
    encoder2 = Conv2D(32, (3, 3), padding='same')(encoder2)
    encoder2 = BatchNormalization()(encoder2)
    encoder2 = Activation('relu')(encoder2)
    encoder2_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder2)
    # 128

    encoder3 = Conv2D(64, (3, 3), padding='same')(encoder2_layer)
    encoder3 = BatchNormalization()(encoder3)
    encoder3 = Activation('relu')(encoder3)
    encoder3 = Conv2D(64, (3, 3), padding='same')(encoder3)
    encoder3 = BatchNormalization()(encoder3)
    encoder3 = Activation('relu')(encoder3)
    encoder3_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder3)
    # 64

    encoder4 = Conv2D(128, (3, 3), padding='same')(encoder3_layer)
    encoder4 = BatchNormalization()(encoder4)
    encoder4 = Activation('relu')(encoder4)
    encoder4 = Conv2D(128, (3, 3), padding='same')(encoder4)
    encoder4 = BatchNormalization()(encoder4)
    encoder4 = Activation('relu')(encoder4)
    encoder4_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder4)
    # 32

    encoder5 = Conv2D(256, (3, 3), padding='same')(encoder4_layer)
    encoder5 = BatchNormalization()(encoder5)
    encoder5 = Activation('relu')(encoder5)
    encoder5 = Conv2D(256, (3, 3), padding='same')(encoder5)
    encoder5 = BatchNormalization()(encoder5)
    encoder5 = Activation('relu')(encoder5)
    encoder5_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder5)
    # 16

    encoder6 = Conv2D(512, (3, 3), padding='same')(encoder5_layer)
    encoder6 = BatchNormalization()(encoder6)
    encoder6 = Activation('relu')(encoder6)
    encoder6 = Conv2D(512, (3, 3), padding='same')(encoder6)
    encoder6 = BatchNormalization()(encoder6)
    encoder6 = Activation('relu')(encoder6)
    encoder6_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder6)
    # 8

    middle = Conv2D(1024, (3, 3), padding='same')(encoder6_layer)
    middle = BatchNormalization()(middle)
    middle = Activation('relu')(middle)
    middle = Conv2D(1024, (3, 3), padding='same')(middle)
    middle = BatchNormalization()(middle)
    middle = Activation('relu')(middle)
    # middle

    decoder6 = UpSampling2D((2, 2))(middle)
    decoder6 = concatenate([encoder6, decoder6], axis=3)
    decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
    decoder6 = BatchNormalization()(decoder6)
    decoder6 = Activation('relu')(decoder6)
    decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
    decoder6 = BatchNormalization()(decoder6)
    decoder6 = Activation('relu')(decoder6)
    decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
    decoder6 = BatchNormalization()(decoder6)
    decoder6 = Activation('relu')(decoder6)
    # 16

    decoder5 = UpSampling2D((2, 2))(decoder6)
    decoder5 = concatenate([encoder5, decoder5], axis=3)
    decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
    decoder5 = BatchNormalization()(decoder5)
    decoder5 = Activation('relu')(decoder5)
    decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
    decoder5 = BatchNormalization()(decoder5)
    decoder5 = Activation('relu')(decoder5)
    decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
    decoder5 = BatchNormalization()(decoder5)
    decoder5 = Activation('relu')(decoder5)
    # 32

    decoder4 = UpSampling2D((2, 2))(decoder5)
    decoder4 = concatenate([encoder4, decoder4], axis=3)
    decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
    decoder4 = BatchNormalization()(decoder4)
    decoder4 = Activation('relu')(decoder4)
    decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
    decoder4 = BatchNormalization()(decoder4)
    decoder4 = Activation('relu')(decoder4)
    decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
    decoder4 = BatchNormalization()(decoder4)
    decoder4 = Activation('relu')(decoder4)
    # 64

    decoder3 = UpSampling2D((2, 2))(decoder4)
    decoder3 = concatenate([encoder3, decoder3], axis=3)
    decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
    decoder3 = BatchNormalization()(decoder3)
    decoder3 = Activation('relu')(decoder3)
    decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
    decoder3 = BatchNormalization()(decoder3)
    decoder3 = Activation('relu')(decoder3)
    decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
    decoder3 = BatchNormalization()(decoder3)
    decoder3 = Activation('relu')(decoder3)
    # 128

    decoder2 = UpSampling2D((2, 2))(decoder3)
    decoder2 = concatenate([encoder2, decoder2], axis=3)
    decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
    decoder2 = BatchNormalization()(decoder2)
    decoder2 = Activation('relu')(decoder2)
    decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
    decoder2 = BatchNormalization()(decoder2)
    decoder2 = Activation('relu')(decoder2)
    decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
    decoder2 = BatchNormalization()(decoder2)
    decoder2 = Activation('relu')(decoder2)
    # 256

    decoder1 = UpSampling2D((2, 2))(decoder2)
    decoder1 = concatenate([encoder1, decoder1], axis=3)
    decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
    decoder1 = BatchNormalization()(decoder1)
    decoder1 = Activation('relu')(decoder1)
    decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
    decoder1 = BatchNormalization()(decoder1)
    decoder1 = Activation('relu')(decoder1)
    decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
    decoder1 = BatchNormalization()(decoder1)
    decoder1 = Activation('relu')(decoder1)
    # 512

    out = Conv2D(num_classes, (1, 1), activation='sigmoid')(decoder1)

    model = Model(inputs=[inputs], outputs=[out])

    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    
    return model



# def build_unet_model(input_shape=(512, 512, 1),
#                  num_classes=1):
#     inputs = Input(shape=input_shape)
#     # 512
    
#     encoder1 = Conv2D(16, (3, 3), padding='same', input_shape=input_shape)(inputs)
#     encoder1 = BatchNormalization()(encoder1)
#     encoder1 = Activation('relu')(encoder1)
#     encoder1 = Conv2D(16, (3, 3), padding='same')(encoder1)
#     encoder1 = BatchNormalization()(encoder1)
#     encoder1 = Activation('relu')(encoder1)
#     encoder1_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder1)
#     # 256

#     encoder2 = Conv2D(32, (3, 3), padding='same')(encoder1_layer)
#     encoder2 = BatchNormalization()(encoder2)
#     encoder2 = Activation('relu')(encoder2)
#     encoder2 = Conv2D(32, (3, 3), padding='same')(encoder2)
#     encoder2 = BatchNormalization()(encoder2)
#     encoder2 = Activation('relu')(encoder2)
#     encoder2_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder2)
#     # 128

#     encoder3 = Conv2D(64, (3, 3), padding='same')(encoder2_layer)
#     encoder3 = BatchNormalization()(encoder3)
#     encoder3 = Activation('relu')(encoder3)
#     encoder3 = Conv2D(64, (3, 3), padding='same')(encoder3)
#     encoder3 = BatchNormalization()(encoder3)
#     encoder3 = Activation('relu')(encoder3)
#     encoder3_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder3)
#     # 64

# #     encoder4 = Conv2D(128, (3, 3), padding='same')(encoder3_layer)
# #     encoder4 = BatchNormalization()(encoder4)
# #     encoder4 = Activation('relu')(encoder4)
# #     encoder4 = Conv2D(128, (3, 3), padding='same')(encoder4)
# #     encoder4 = BatchNormalization()(encoder4)
# #     encoder4 = Activation('relu')(encoder4)
# #     encoder4_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder4)
# #     # 32

# #     encoder5 = Conv2D(256, (3, 3), padding='same')(encoder4_layer)
# #     encoder5 = BatchNormalization()(encoder5)
# #     encoder5 = Activation('relu')(encoder5)
# #     encoder5 = Conv2D(256, (3, 3), padding='same')(encoder5)
# #     encoder5 = BatchNormalization()(encoder5)
# #     encoder5 = Activation('relu')(encoder5)
# #     encoder5_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder5)
# #     # 16

# #     encoder6 = Conv2D(512, (3, 3), padding='same')(encoder5_layer)
# #     encoder6 = BatchNormalization()(encoder6)
# #     encoder6 = Activation('relu')(encoder6)
# #     encoder6 = Conv2D(512, (3, 3), padding='same')(encoder6)
# #     encoder6 = BatchNormalization()(encoder6)
# #     encoder6 = Activation('relu')(encoder6)
# #     encoder6_layer = MaxPooling2D((2, 2), strides=(2, 2))(encoder6)
# #     # 8

#     middle = Conv2D(128, (3, 3), padding='same')(encoder3_layer)
#     middle = BatchNormalization()(middle)
#     middle = Activation('relu')(middle)
#     middle = Conv2D(128, (3, 3), padding='same')(middle)
#     middle = BatchNormalization()(middle)
#     middle = Activation('relu')(middle)
#     # middle

# #     decoder6 = UpSampling2D((2, 2))(middle)
# #     decoder6 = concatenate([encoder6, decoder6], axis=3)
# #     decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
# #     decoder6 = BatchNormalization()(decoder6)
# #     decoder6 = Activation('relu')(decoder6)
# #     decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
# #     decoder6 = BatchNormalization()(decoder6)
# #     decoder6 = Activation('relu')(decoder6)
# #     decoder6 = Conv2D(512, (3, 3), padding='same')(decoder6)
# #     decoder6 = BatchNormalization()(decoder6)
# #     decoder6 = Activation('relu')(decoder6)
#     # 16

# #     decoder5 = UpSampling2D((2, 2))(decoder6)
# #     decoder5 = concatenate([encoder5, decoder5], axis=3)
# #     decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
# #     decoder5 = BatchNormalization()(decoder5)
# #     decoder5 = Activation('relu')(decoder5)
# #     decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
# #     decoder5 = BatchNormalization()(decoder5)
# #     decoder5 = Activation('relu')(decoder5)
# #     decoder5 = Conv2D(256, (3, 3), padding='same')(decoder5)
# #     decoder5 = BatchNormalization()(decoder5)
# #     decoder5 = Activation('relu')(decoder5)
# #     # 32

# #     decoder4 = UpSampling2D((2, 2))(decoder5)
# #     decoder4 = concatenate([encoder4, decoder4], axis=3)
# #     decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
# #     decoder4 = BatchNormalization()(decoder4)
# #     decoder4 = Activation('relu')(decoder4)
# #     decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
# #     decoder4 = BatchNormalization()(decoder4)
# #     decoder4 = Activation('relu')(decoder4)
# #     decoder4 = Conv2D(128, (3, 3), padding='same')(decoder4)
# #     decoder4 = BatchNormalization()(decoder4)
# #     decoder4 = Activation('relu')(decoder4)
# #     # 64

#     decoder3 = UpSampling2D((2, 2))(middle)
#     decoder3 = concatenate([encoder3, decoder3], axis=3)
#     decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
#     decoder3 = BatchNormalization()(decoder3)
#     decoder3 = Activation('relu')(decoder3)
#     decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
#     decoder3 = BatchNormalization()(decoder3)
#     decoder3 = Activation('relu')(decoder3)
#     decoder3 = Conv2D(64, (3, 3), padding='same')(decoder3)
#     decoder3 = BatchNormalization()(decoder3)
#     decoder3 = Activation('relu')(decoder3)
#     # 128

#     decoder2 = UpSampling2D((2, 2))(decoder3)
#     decoder2 = concatenate([encoder2, decoder2], axis=3)
#     decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
#     decoder2 = BatchNormalization()(decoder2)
#     decoder2 = Activation('relu')(decoder2)
#     decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
#     decoder2 = BatchNormalization()(decoder2)
#     decoder2 = Activation('relu')(decoder2)
#     decoder2 = Conv2D(32, (3, 3), padding='same')(decoder2)
#     decoder2 = BatchNormalization()(decoder2)
#     decoder2 = Activation('relu')(decoder2)
#     # 256

#     decoder1 = UpSampling2D((2, 2))(decoder2)
#     decoder1 = concatenate([encoder1, decoder1], axis=3)
#     decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
#     decoder1 = BatchNormalization()(decoder1)
#     decoder1 = Activation('relu')(decoder1)
#     decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
#     decoder1 = BatchNormalization()(decoder1)
#     decoder1 = Activation('relu')(decoder1)
#     decoder1 = Conv2D(16, (3, 3), padding='same')(decoder1)
#     decoder1 = BatchNormalization()(decoder1)
#     decoder1 = Activation('relu')(decoder1)
#     # 512

#     out = Conv2D(num_classes, (1, 1), activation='sigmoid')(decoder1)

#     model = Model(inputs=[inputs], outputs=[out])

#     model.compile(optimizer=RMSprop(learning_rate=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    
#     return model



def classify(model, data, modelCompile = False, batch_size = 64, epoch = 100):
    
    if modelCompile == True:
        model = build_unet_model()
    
    trainA =None
    valA = None
    testA = None
    
    
    trainA, valA, testA = data[0], data[1], data[2]

    callbacks = [EarlyStopping(patience=2, monitor = 'loss', verbose=1)]
#     train_DataUnet = tf.data.Dataset.from_tensor_slices((trainA[0], trainA[1]))    
#     train_DataUnet = train_DataUnet.shuffle(buffer_size=1024).batch(64)

#     # Prepare the validation dataset
#     val_DataUnet = tf.data.Dataset.from_tensor_slices((valA[0], valA[1]))
#     val_DataUnet = val_DataUnet.batch(64)
    
    
    resultsA = model.fit(trainA[0], trainA[1], batch_size=batch_size, epochs=epoch, callbacks=callbacks, validation_data=(valA[0], valA[1]))
    
    predictionsA = model.predict(testA[0])
    
    model.evaluate(testA[0], testA[1])
    
    predictionsA[predictionsA >= 0.5 ] =1.0
    predictionsA[predictionsA < 0.5 ] = 0.0    
    
    return resultsA, predictionsA, model
    


# def trainUnet(trainX, trainY, valX, valY, batch_size = 32, epoch = 100):

# #     model = build_unet_model()


#     callbacks = [
#             EarlyStopping(patience=5, monitor = 'loss', verbose=1),
#             ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
#             ModelCheckpoint('model-circle.h5', verbose=1, save_best_only=True, save_weights_only=True)
#             ]

#     results = model.fit(trainX, trainY, batch_size=batch_size, epochs=epoch, callbacks=callbacks, validation_data=(valX, valY))
    
#     return results, model