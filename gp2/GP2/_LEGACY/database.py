import numpy as np
import modelUnet
import modelCNN
from datapoint import Datapoint

class Data():
    file_path = None
    trained_modelA = None
    trained_modelC = None
    trainA = None
    valA = None
    testA = None
    human = None

    predictionsA = None

    testC_index = None
    predictionsC = None

    new_trainA = None
    new_testA = None
    historyA = {"loss": [],
                "accuracy": []}

    @staticmethod
    def get(stringValue, file_path=None):

        if file_path is None:
            Data.file_path = "/home/neha.goyal001/deepHealth/gp2/dataset"
        else:
            Data.file_path = str(file_path)

        if stringValue == 'A':
            # original noised images and labels
            labels = np.load(Data.file_path + "/data/Labels20.npy")
            noisedImages = np.load(Data.file_path + "/data/noisedImages20.npy")
            imageData = Datapoint(noisedImages, labels)
            # get the images and their corresponding labels in 4 datasets
            # dataset1: trainA for training the classifier
            # dataset2: valA for validating the classifier
            # dataset3: testA for predictions (using classifier)
            # dataset4: human is expert dataset
            Data.trainA, Data.valA, Data.testA, Data.human = modelUnet.classifierDataset(imageData)

            # datasets required for classifier
            dataA = (Data.trainA, Data.valA, Data.testA)

            return dataA
        elif stringValue == 'B':
            # expert generated dataset

            return Data.human
        elif stringValue == 'C':
            # test A is the classifier test dataset
            # test_images, test_labels = testA[0], testA[1]
            # test_predictions = predictionsA
            testDataset = Data.testA
            testPredictions = Data.predictionsA
            machine = (testDataset[0], testPredictions)

            # machine + human generated dataset
            trainC, valC, testC, Data.testC_index = modelCNN.discriminatorDataset(machine[0], machine[1], Data.human[0],
                                                                                  Data.human[1])
            dataC = (trainC, valC, testC)
            return dataC
        else:
            print("Incorrect data selection")
            return None

    def find_in_repo(self, unique_id):
        Datapoint.data[unique_id][2]

    def move_from_repo(self, unique_id, new_repo):
        Datapoint.data[unique_id][2] = new_repo
