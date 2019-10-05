## HUMAN ACTIVITY RECOGNITION
## AUSTIN KOENIG


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

#from keras import models
#from keras import layers
#from keras import optimizers
from keras.preprocessing import sequence

from sklearn import preprocessing

class HARSystem(object):
    '''
    Human Activity Recognition (HAR) System class.

    Methods
    -------
    preprocess_data : None
        Loads, splits, and scales data.

    resample_classes : None
        Resamples data to reduce class size imbalance.

    build_model : None
        Builds model.

    evaluate_model : None
        Train and evaluates model.

    '''
    def __init__(self):
        self.data_fp = './data/' # data filepath
        self.tr_split = 0.6 # train split
        self.val_ts_split = 0.2 # val/test split
        self.sscaler = [preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False) for _ in range(3)] # scalers for each channel
        self.lscaler = [preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False) for _ in range(3)]

        self.models = {}

        self.labels = None
        self.raw_data = None

        self.data = {
            'x': {
                'ss': np.empty((0, 3, 256)),
                'ls': np.empty((0, 3, 10000))
            },
            'y': None
        }

        self.train = {
            'x': {
                'ss': np.empty((0, 3, 256)),
                'ls': np.empty((0, 3, 10000))
            },
            'y': None
        }

        self.test = {
            'x': {
                'ss': np.empty((0, 3, 256)),
                'ls': np.empty((0, 3, 10000))
            },
            'y': None
        }

        self.val = {
            'x': {
                'ss': np.empty((0, 3, 256)),
                'ls': np.empty((0, 3, 10000))
            },
            'y': None
        }

    def preprocess_data(self):
        '''
        Prepares the data for the activity recognizer.
        '''
        
        print("PREPROCESSING DATA...")

        self.labels = os.listdir(self.data_fp) # list of file names to be indexed by output label
        rawX = []
        rawY = []
        class_samples = [] # number of samples in each class
        total_samples = 0 # total number of samples

        # DATA EXTRACTION
        print("Extracting data from files...")
        for i in range(len(self.labels)):
            root = os.path.join(self.data_fp, self.labels[i])
            files = os.listdir(root) # go through data directories
            class_samples.append(0)

            for f in files:
                x = np.genfromtxt(os.path.join(root, f), delimiter = ' ').T # get sample from text file
                rawX.append(x) # raw sample input
                rawY.append(i) # raw sample output

                xs = sequence.pad_sequences(x, maxlen = 256, value = -1) # short-sequence samples
                xl = sequence.pad_sequences(x, maxlen = 10000, value = -1) # long-sequence samples

                # reshape arrays to be stacked
                xs = np.reshape(xs, (1, xs.shape[0], xs.shape[1]))
                xl = np.reshape(xl, (1, xl.shape[0], xl.shape[1]))

                self.data['x']['ss'] = np.vstack((self.data['x']['ss'], xs)) # vertically stack samples
                self.data['x']['ls'] = np.vstack((self.data['x']['ls'], xl))

                class_samples[i] += 1
                total_samples += 1

        self.data['y'] = np.array(rawY)
        self.data['y'] = np.array(rawY)
        self.raw_data = (rawX, rawY) # copy of raw data
        print("Data extraction complete.")

        # DATA SPLITTING
        print("Splitting data...")

        train_idx = round(self.tr_split * self.data['x']['ss'].shape[0]) # train index
        val_idx = round(self.val_ts_split * self.data['x']['ss'].shape[0]) # val/test index

        self.train['x']['ss'] = self.data['x']['ss'][:train_idx, :, :]
        self.train['x']['ls'] = self.data['x']['ls'][:train_idx, :, :]
        self.train['y'] = self.data['y'][:train_idx]

        self.val['x']['ss'] = self.data['x']['ss'][train_idx:(train_idx + val_idx), :, :]
        self.val['x']['ls'] = self.data['x']['ls'][train_idx:(train_idx + val_idx), :, :]
        self.val['y'] = self.data['y'][train_idx:(train_idx + val_idx)]

        self.test['x']['ss'] = self.data['x']['ss'][(train_idx + val_idx):, :, :]
        self.test['x']['ls'] = self.data['x']['ls'][(train_idx + val_idx):, :, :]
        self.test['y'] = self.data['y'][(train_idx + val_idx):]

        print("    Short Sequence Shapes:")
        print(f"        X Train: {self.train['x']['ss'].shape}")
        print(f"        Y Train: {self.train['y'].shape}")
        print(f"        X Validation: {self.val['x']['ss'].shape}")
        print(f"        Y Validation: {self.val['y'].shape}")
        print(f"        X Test: {self.test['x']['ss'].shape}")
        print(f"        Y Test: {self.test['y'].shape}")
        print("    Long Sequence Shapes:")
        print(f"        X Train: {self.train['x']['ls'].shape}")
        print(f"        Y Train: {self.train['y'].shape}")
        print(f"        X Validation: {self.val['x']['ls'].shape}")
        print(f"        Y Validation: {self.val['y'].shape}")
        print(f"        X Test: {self.test['x']['ls'].shape}")
        print(f"        Y Test: {self.test['y'].shape}")
        print("Data splitting complete.")

        # DATA SCALING
        print("Scaling data...")
        print("    Saving scalers...")
        for i in range(3):
            # numpy forces us to scale each channel individually, thus
            # needing six total scalers
            self.sscaler[i].fit_transform(self.train['x']['ss'][:, i, :])
            self.lscaler[i].fit_transform(self.train['x']['ls'][:, i, :])

            self.sscaler[i].transform(self.val['x']['ss'][:, i, :])
            self.lscaler[i].transform(self.val['x']['ls'][:, i, :])

            self.sscaler[i].transform(self.test['x']['ss'][:, i, :])
            self.lscaler[i].transform(self.test['x']['ls'][:, i, :])

            ssfn = f'./scalers/sscaler_{i + 1}.sc'
            lsfn = f'./scalers/lscaler_{i + 1}.sc'

            # save scalers to disk
            joblib.dump(self.sscaler[i], ssfn)
            joblib.dump(self.lscaler[i], lsfn)

            print(f"        Saved sscaler to {ssfn}")
            print(f"        Saved lscaler to {lsfn}")
        print("    Scalers saved.")
        print("Data scaling complete.")
        print("PREPROCESSING COMPLETE.")

    def resample_classes(self):
        pass
    
    def build_model(self):
        pass

    def evaluate_model(self):
        pass



if __name__ == "__main__":
    rec = HARSystem()
    rec.preprocess_data()



