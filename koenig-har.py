## HUMAN ACTIVITY RECOGNITION
## AUSTIN KOENIG


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import shutil
import joblib

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras.preprocessing import sequence

from sklearn import preprocessing
from sklearn import metrics
from sklearn import utils

np.random.seed(12)
sys.stdout = open('output_log.txt', 'w')

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
        self.sequence_length = 1024
        self.scaler = [preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False) for _ in range(3)] # scalers for each channel

        self.MODELS = {}
        self.HISTS = {}
        self.TESTS = {}

        self.labels = None
        self.raw_data = None

        self.data = {
            'x': np.empty((0, self.sequence_length, 3)),
            'y': None
        }

        self.train = {
            'x': np.empty((0, self.sequence_length, 3)),
            'y': None
        }

        self.test = {
            'x': np.empty((0, self.sequence_length, 3)),
            'y': None
        }

        self.val = {
            'x': np.empty((0, self.sequence_length, 3)),
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
        print("    Extracting data from files...")
        for i in range(len(self.labels)):
            root = os.path.join(self.data_fp, self.labels[i])
            files = os.listdir(root) # go through data directories
            class_samples.append(0)

            for f in files:
                x = np.genfromtxt(os.path.join(root, f), delimiter = ' ') # get sample from text file
                rawX.append(x) # raw sample input
                rawY.append(i) # raw sample output

                xx = sequence.pad_sequences(x.T, maxlen = self.sequence_length, value = -1) # short-sequence samples                
                xx = np.reshape(xx, (1, xx.shape[1], xx.shape[0])) # reshape arrays to be stacked
                self.data['x'] = np.vstack((self.data['x'], xx)) # vertically stack samples

                class_samples[i] += 1
                total_samples += 1

        y = np.array(rawY)
        n_cats = np.max(y) + 1
        y = keras.utils.to_categorical(y, n_cats) # one-hot encode outputs
        new_indices = np.arange(self.data['x'].shape[0])
        np.random.shuffle(new_indices) # shuffle data
        self.data['x'] = self.data['x'][new_indices, :, :]
        self.data['y'] = y[new_indices, :]
        self.raw_data = (rawX, rawY) # copy of raw data
        print("    Data extraction complete.")

        # DATA SPLITTING
        print("    Splitting data...")

        train_idx = round(self.tr_split * self.data['x'].shape[0]) # train index
        val_idx = round(self.val_ts_split * self.data['x'].shape[0]) # val/test index

        self.train['x'] = self.data['x'][:train_idx, :, :]
        self.train['y'] = self.data['y'][:train_idx, :]

        self.val['x'] = self.data['x'][train_idx:(train_idx + val_idx), :, :]
        self.val['y'] = self.data['y'][train_idx:(train_idx + val_idx), :]

        self.test['x'] = self.data['x'][(train_idx + val_idx):, :, :]
        self.test['y'] = self.data['y'][(train_idx + val_idx):, :]

        print("        Sequence Shapes:")
        print(f"            X Train: {self.train['x'].shape}")
        print(f"            Y Train: {self.train['y'].shape}")
        print(f"            X Validation: {self.val['x'].shape}")
        print(f"            Y Validation: {self.val['y'].shape}")
        print(f"            X Test: {self.test['x'].shape}")
        print(f"            Y Test: {self.test['y'].shape}")
        print("    Data splitting complete.")

        # DATA SCALING
        print("    Scaling data...")
        print("        Saving scalers...")
        for i in range(3):
            # numpy forces us to scale each channel individually, thus
            # needing three total scalers
            self.scaler[i].fit_transform(self.train['x'][:, i, :])
            self.scaler[i].transform(self.val['x'][:, i, :])
            self.scaler[i].transform(self.test['x'][:, i, :])
            sfn = f'./files/scalers/scaler_{i + 1}'
            joblib.dump(self.scaler[i], sfn) # save scalers to disk
            print(f"            Saved scaler_{i + 1} to {sfn}")
        print("        Scalers saved.")
        print("    Data scaling complete.")
        print("    Saving processed data...")
        joblib.dump(self.data, './files/data/processed_data')
        print("    Processed data saved to ./files/data/processed_data")
        print("PREPROCESSING COMPLETE.")
    
    def cnn(self):
        model = models.Sequential([
            layers.Conv1D(128, 8, activation = 'relu', input_shape = self.data['x'].shape[1:]),
            layers.Conv1D(128, 8, activation = 'relu'),
            layers.Conv1D(128, 8, activation = 'relu'),
            layers.MaxPooling1D(8),
            layers.Dropout(0.5),
            layers.Conv1D(256, 6, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Conv1D(256, 6, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Conv1D(256, 6, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.MaxPooling1D(6),
            layers.Dropout(0.5),
            layers.Conv1D(256, 4, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Conv1D(256, 4, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Conv1D(256, 4, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.MaxPooling1D(4),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(128, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Dense(128, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(len(self.labels), activation = 'softmax')
        ])
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(), metrics = ['acc'])
        model.summary()
        return model
    
    def generate_models(self):
        print("GENERATING MODELS...")
        #self.MODELS['GRU'] = self.build_GRU_model()
        self.MODELS['Conv1D'] = self.cnn()
        print("MODEL GENERATION COMPLETE.")

    def evaluate_models(self):
        print("EVALUATING MODELS...")
        EPCHS = 300
        BATCH = 16

        print("    Training/testing models...")
        for k in self.MODELS:
            print(f"        Current Model: {k}")
            self.HISTS[k] = self.MODELS[k].fit(self.train['x'], self.train['y'], 
                                               epochs = EPCHS, batch_size = BATCH, 
                                               validation_data = (self.val['x'], self.val['y']), 
                                               verbose = 2).history
            self.TESTS[k] = self.MODELS[k].evaluate(self.test['x'], self.test['y'], verbose = 1)
            print(f"            Model {k} Test Loss: {self.TESTS[k][0]}")
            print(f"            Model {k} Test Accuracy: {self.TESTS[k][1]}")
            print(f"        Model {k} evaluation complete.")
        print("    Training/testing complete.")
        print("    Generating and saving plots/metrics...")
        
        for k in self.HISTS:
            loss_figure = plt.figure(figsize = (20, 15))
            al = loss_figure.add_subplot(111)
            acc_figure = plt.figure(figsize = (20, 15))
            aa = acc_figure.add_subplot(111)

            al.plot(range(EPCHS), self.HISTS[k]['loss'], label = f'{k} Training')
            al.plot(range(EPCHS), self.HISTS[k]['val_loss'], label = f'{k} Validation')
            al.plot(np.repeat(self.TESTS[k][0], EPCHS), label = f'{k} Testing')

            aa.plot(range(EPCHS), self.HISTS[k]['acc'], label = f'{k} Training')
            aa.plot(range(EPCHS), self.HISTS[k]['val_acc'], label = f'{k} Validation')
            aa.plot(np.repeat(self.TESTS[k][1], EPCHS), label = f'{k} Testing')
        
            al.title.set_text("Model Losses")
            al.set_xlabel("Epoch")
            al.set_ylabel("Loss")
            al.legend()

            aa.title.set_text("Model Accuracies")
            aa.set_xlabel("Epoch")
            aa.set_ylabel("Accuracy")
            aa.legend()

            preds = self.MODELS[k].predict(self.test['x'])
            predy = np.argmax(preds, axis = 1)
            truey = np.argmax(self.test['y'], axis = 1)
            labs = np.arange(len(self.labels))[utils.multiclass.unique_labels(predy)]
            cm = metrics.confusion_matrix(truey, predy, labels = labs)
            cr = metrics.classification_report(truey, predy)

            conf_mat_fig = plt.figure(figsize = (20, 15))
            ac = conf_mat_fig.add_subplot(111)
            im = ac.matshow(cm)
            conf_mat_fig.colorbar(im)
            # ac.set_xticklabels(labs)
            # ac.set_yticklabels(labs)
            ac.set_xlabel('Predicted')
            ac.set_ylabel('True')

            conf_mat_fig.savefig(f'./files/figures/{k}-confusion_matrix.pdf')
            joblib.dump(cr, f'./files/metrics/{k}-classification_report')
            joblib.dump(cm, f'./files/metrics/{k}-confusion_matrix')

            print(f"\n    {k} Classification Report: ")
            print(cr)
            print(f"\n    {k} Confusion Matrix: ")
            print(cm)

            loss_figure.savefig('./files/figures/loss.pdf')
            acc_figure.savefig('./files/figures/acc.pdf')
            print("    Plots/metrics generated and saved.")
        print("MODEL EVALUATION COMPLETE.")

    def create_dirs(self):
        dirs = ['./files/', './files/figures/', './files/data/', './files/models/', './files/metrics/', './files/models/gru/', './files/models/rnn/', './files/scalers/']
        self.clean_dirs()
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
    
    def clean_dirs(self):
        if os.path.exists('./files/'):
            shutil.rmtree('./files/')
    
    def execute(self):
        self.create_dirs()
        self.preprocess_data()
        self.generate_models()
        self.evaluate_models()



if __name__ == "__main__":
    rec = HARSystem()
    rec.execute()



