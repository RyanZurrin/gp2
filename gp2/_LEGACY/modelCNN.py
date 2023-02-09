import numpy as np
import split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, LeakyReLU, Dropout, Flatten, Dense







def discriminatorDataset(machine_image, machine_label, human_image, human_label):
    
#     machine_image, machine_label = machine[0], machine[1]
#     human_image, human_label = human[0], human[1]
    
    # create label class for machine and expert predicted images
    machine_y, human_y = np.ones(machine_image.shape[0], dtype = int), np.zeros(human_image.shape[0], dtype = int)
    machine_image = np.asarray(machine_image, dtype=np.float32)
    human_image = np.asarray(human_image, dtype=np.float32)

    # normalize
    machine_image = machine_image / np.max(machine_image)
    human_image = human_image / np.max(human_image)
    
    
    # Concatenate both machine and human dataset for training and testing
    classifyImage = np.concatenate((machine_image, human_image), axis = 0)
    classifyLabel = np.concatenate((machine_label, human_label), axis = 0)    
    classify_Y = np.concatenate((machine_y, human_y), axis = 0)
    classify_Y = to_categorical(classify_Y)
    
    
    n_vals, perm  = split.dataSplit(classifyImage.shape[0], 3)
    
    trainC_image, valC_image, testC_image = classifyImage[perm[0:n_vals[0]], ...], classifyImage[perm[n_vals[0]:n_vals[1]], ...], classifyImage[perm[n_vals[1]:n_vals[2]], ...]
    
    trainC_label, valC_label, testC_label =  classifyLabel[perm[0:n_vals[0]], ...], classifyLabel[perm[n_vals[0]:n_vals[1]], ...], classifyLabel[perm[n_vals[1]:n_vals[2]], ...]
    
    trainC_y, valC_y, testC_y = classify_Y[perm[0:n_vals[0]]], classify_Y[perm[n_vals[0]:n_vals[1]]], classify_Y[perm[n_vals[1]:n_vals[2]]]
    
    id_array = perm[n_vals[1]:n_vals[2]]
    
    testC_index = {i : id_array[i] for i in range(len(id_array)) if id_array[i] < machine_image.shape[0]} 
    
    
    return (trainC_image, trainC_label, trainC_y), (valC_image, valC_label, valC_y), (testC_image, testC_label, testC_y), testC_index



def create_convolution_layers(input_img, input_shape):
    model = Conv2D(16, (3, 3), padding='same', input_shape=input_shape)(input_img)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D((2, 2),padding='same')(model)
    model = Dropout(0.25)(model)
    #256
    
    model = Conv2D(32, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D((2, 2),padding='same')(model)
    model = Dropout(0.25)(model)
    #128

    model = Conv2D(64, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
    model = Dropout(0.25)(model)
    #64

    model = Conv2D(128, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
    model = Dropout(0.4)(model)
    #32
    
    model = Conv2D(256, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
    model = Dropout(0.4)(model)
    #16

    return model


def build_cnn_model(Image_Shape1 = (512, 512, 1), Image_Shape2 = (512, 512, 1), num_classes = 2):
    
    image_input = Input(shape=Image_Shape1)
    #512
    image_model = create_convolution_layers(image_input, Image_Shape1)
    
    
    label_input = Input(shape=Image_Shape2)
    #512
    label_model = create_convolution_layers(label_input, Image_Shape2)
    
    
    conv = concatenate([image_model, label_model])

    conv = Flatten()(conv)

    dense = Dense(512)(conv)
    dense = LeakyReLU(alpha=0.1)(dense)
    dense = Dropout(0.5)(dense)

    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[image_input, label_input], outputs=[output])

    # compile the model
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model


def discriminator(model, data, modelCompile = False, batch_size=64, epoch = 100):
    
    
    # create the train and test dataset for discriminator
    # here: trainB_image, trainB_label, trainB_y = trainB[0], trainB[1], trainB[2] 
    # here: testB_image, testB_label, testB_y = testB[0], testB[1], testB[2]
    
    
#     trainC, valC, testC, testC_index = discriminatorDataset(machine[0], machine[1], human[0], human[1])
    trainC, valC, testC = data[0], data[1], data[2]
    callbacks = [EarlyStopping(patience=3, monitor = 'loss', verbose=1)]
    
    
    if modelCompile == True:
        model = build_cnn_model()
        
        # train_Data = tf.data.Dataset.from_tensor_slices(((trainB_image, trainB_label), trainB_y))    
        # train_Data = train_Data.shuffle(buffer_size=1024).batch(64)
        # results = model.fit(train_Data, epochs=epoch, callbacks=callbacks)   

        # train the discriminator model
        model.fit(x = [trainC[0], trainC[1]], y = trainC[2], epochs=epoch, batch_size=batch_size, callbacks=callbacks, validation_data=([valC[0], valC[1]], valC[2]))
    
    
    

    
    # predictions using discriminator model
    predictionsC = model.predict(x = [testC[0], testC[1]])
    model.evaluate(x = [testC[0], testC[1]], y = testC[2])
            
    return predictionsC, testC[1], testC[2], model
    
    
    
