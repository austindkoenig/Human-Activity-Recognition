## HUMAN ACTIVITY RECOGNITION
## AUSTIN KOENIG


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#from keras import models
#from keras import layers
#from keras import optimizers
#from keras.preprocessing import sequence

class HARSystem(object):
    '''
    Human Activity Recognition (HAR) System class.
    '''
    def __init__(self, data_fp = './data/'):
        self.data_fp = data_fp
        self.scaler = None

        self.labels = None
        self.raw_data = None
        self.data = None

        self.train = None
        self.val = None
        self.test = None
    
    def preprocess_data(self):
        '''
        Prepares the data for the activity recognizer.

        Returns
        -------
        trainX : numpy.array
            Training sequences
        '''

        rawX = []
        rawY = []

        self.labels = os.listdir(self.data_fp)

        # Extract data from text files
        for i in range(len(self.labels)):
            root = os.path.join(self.data_fp, self.labels[i])
            files = os.listdir(root)
            class_samples = 0

            for f in files:
                x = np.genfromtxt(os.path.join(root, f), delimiter = ' ')
                rawX.append(x)
                rawY.append(i)
                class_samples += 1
            
            min_length = min([len(x) for x in rawX])
            max_length = max([len(x) for x in rawX])
            print(f'| {self.labels[i]} | {class_samples} | {round(class_samples / 839, 6)} | {min_length} | {max_length} |')
        print()
        
        print(len(rawX))
        #rawX = np.reshape(np.dstack(rawX), (len(rawX), 3, -1))
        #rawY = np.array(rawY)
        #print(rawX.shape, rawY.shape)
        #self.raw_data = (rawX, rawY)

    def build_model(self):
        pass

    def evaluate_model(self):
        pass



if __name__ == "__main__":
    rec = HARSystem()
    rec.preprocess_data()



