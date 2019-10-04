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
from keras.preprocessing import sequence

class HARSystem(object):
    '''
    Human Activity Recognition (HAR) System class.
    '''
    def __init__(self, data_fp = './data/'):
        self.data_fp = data_fp
        self.scaler = None

        self.labels = None
        self.raw_data = None
        self.data = {
            'ss': {
                'x': np.empty((0, 3, 256)),
                'y': None
            },
            'ls': {
                'x': np.empty((0, 3, 10000)),
                'y': None
            }
        }

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

        class_samples = []
        total_samples = 0

        # Extract data from text files
        for i in range(len(self.labels)):
            root = os.path.join(self.data_fp, self.labels[i])
            files = os.listdir(root)
            class_samples.append(0)

            for f in files:
                x = np.genfromtxt(os.path.join(root, f), delimiter = ' ').T
                rawX.append(x)
                rawY.append(i)

                xs = sequence.pad_sequences(x, maxlen = 256, value = -1)
                xl = sequence.pad_sequences(x, maxlen = 10000, value = -1)
                xs = np.reshape(xs, (1, xs.shape[0], xs.shape[1]))
                xl = np.reshape(xl, (1, xl.shape[0], xl.shape[1]))

                self.data['ss']['x'] = np.vstack((self.data['ss']['x'], xs))
                self.data['ls']['x'] = np.vstack((self.data['ls']['x'], xl))

                class_samples[i] += 1
                total_samples += 1
            #min_length = min([len(x) for x in rawX])
            #max_length = max([len(x) for x in rawX])
            #print(f'| {self.labels[i]} | {class_samples} | {round(class_samples / 839, 6)} | {min_length} | {max_length} |')

        self.data['ss']['y'] = np.array(rawY)
        self.data['ls']['y'] = np.array(rawY)
        self.raw_data = (rawX, rawY)
        print(self.data['ss']['x'].shape, self.data['ss']['y'].shape)

    def build_model(self):
        pass

    def evaluate_model(self):
        pass



if __name__ == "__main__":
    rec = HARSystem()
    rec.preprocess_data()



