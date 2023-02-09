import split
import numpy as np
import os
import modelUnet
import modelCNN
import tensorflow as tf
from tensorflow import keras
import pickle

class Data():  
        
    file_path = None     
    trained_modelA = None
    trained_modelC = None
    trainA = None
    valA = None   
    testA = None
    dataB = None
    
        
    predictionsA = None
    
    testC_index = None        
    predictionsC = None


    historyA = {"loss": [], 
               "accuracy": []}
    
    

        
    @staticmethod  
    def get(stringValue, file_path = None ):    
        
        
        if file_path == None:
#             Data.file_path = str(os.getcwd())
            Data.file_path = "/home/neha.goyal001/deepHealth/gp2/dataset"
        else:
            Data.file_path = str(file_path)        
        
        if stringValue == 'A':         
            
            # original noised images and labels
            labels = np.load(Data.file_path+"/data/Labels20.npy")
            noisedImages = np.load(Data.file_path+"/data/noisedImages20.npy")
            
            # get the images and their corresponding labels in 4 datasets
            # dataset1: trainA for training the classifier
            # dataset2: valA for validating the classifier
            # dataset3: testA for predictions (using classifier)
            # dataset4: human is expert dataset
            Data.trainA, Data.valA, Data.testA, Data.dataB = modelUnet.classifierDataset(noisedImages, labels)

            # datasets required for classifier
            dataA = (Data.trainA, Data.valA, Data.testA)
            

            return dataA
        
        if stringValue == 'B': 
            
            #expert generated dataset
            
            return Data.dataB
            
        if stringValue == 'C':
            
            # test A is the classifier test dataset
            # test_images, test_labels = testA[0], testA[1] 
            # test_predictions = predictionsA
            testDataset = Data.testA
            testPredictions = Data.predictionsA
            machine = (testDataset[0], testPredictions)
            
            #machine + human generated dataset
            trainC, valC, testC, Data.testC_index = modelCNN.discriminatorDataset(machine[0], machine[1], Data.dataB[0], Data.dataB[1])       
            dataC = (trainC, valC, testC)
            return dataC

            
        
        
        

def train_classifier(dataA, modelCompile = False):
    
    #load model
    
    if modelCompile == False :
        print("Using trained classifier model")

        model =  Data.trained_modelA
    else:
        model = None
    
     # Use dataset 'dataA' to get training results and predictions
    resultsA, Data.predictionsA, Data.trained_modelA = modelUnet.classify(model, dataA, modelCompile)
    
    Data.historyA["loss"].append(resultsA.history['loss'][0])
    Data.historyA["accuracy"].append(resultsA.history['dice_coeff'][0])
    

def train_discriminator(dataC, modelCompile = False):
    
    if modelCompile == False:
        print("Using trained discriminator model")

        model = Data.trained_modelC
    else:
        model = None
     
    Data.predictionsC, testC_label, testC_y, Data.trained_modelC = modelCNN.discriminator(model, dataC, modelCompile)   
    


    
def find_machine_labels():
    req_index = []

    print("Old Train Shape: " + str(Data.trainA[0].shape) + str(Data.trainA[1].shape))
    print("Old Test Shape: " + str(Data.testA[0].shape) + str(Data.testA[1].shape))
    

    for index_key, index_value in Data.testC_index.items():
        if Data.predictionsC[index_key][1]==1:
            req_index.append(index_value)
            

    
    extra_add = req_index[0:int(len(req_index)/8)]

    retain = [index_value for index_value in range(Data.testA[0].shape[0]) if index_value not in extra_add]
 
    new_trainA = (np.concatenate((Data.trainA[0], Data.testA[0][extra_add]), axis = 0), np.concatenate((Data.trainA[1], Data.testA[1][extra_add]), axis = 0))
    new_testA = (Data.testA[0][retain], Data.testA[1][retain])
    print("Total index: " + str(len(req_index)))
    print("New Train Shape: " + str(new_trainA[0].shape) + str(new_trainA[1].shape))
    print("New Test Shape: " + str(new_testA[0].shape) + str(new_testA[1].shape))
    
    dataD = (new_trainA, Data.valA, new_testA)
    Data.trainA = new_trainA
    Data.testA = new_testA

    
    return dataD
