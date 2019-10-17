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

from sklearn import preprocessing
from sklearn import metrics
from sklearn import utils

np.random.seed(12)
sys.stdout = open('output_log.txt', 'w')

class HARProject(object):
    '''
    Human Activity Recognition (HAR) Project class.

    Usage
    -----
        proj = HARProject()
        proj.preprocess_data()
        proj.generate_models()
        proj.evaluate_models()

    '''
    def __init__(self):
        self.data_fp = '' # data filepath
        self.sequence_length = 1024

        self.class_frequencies = {}

        self.scalers = {
            'x': preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False),
            'y': preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False),
            'z': preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False)
        }

        self.classifiers = {}
        self.histories = {}
        self.tests = {}

        # self.labels = None
        # self.raw_data = None

        self.data = {
            'labels': None,
            'raw': None,

            'all': {
                'x': np.empty((0, self.sequence_length, 3)),
                'y': None
            },

            'train': {
                'x': np.empty((0, self.sequence_length, 3)),
                'y': None
            },

            'validation': {
                'x': np.empty((0, self.sequence_length, 3)),
                'y': None
            },

            'test': {
                'x': np.empty((0, self.sequence_length, 3)),
                'y': None
            }
        }

    def preprocess_data(self, filepath = './data/', split = (0.7, 0.15)):
        '''
        Prepares the data for the activity recognizer.
        '''
        
        print("PREPROCESSING DATA...")
        self.data_fp = filepath
        self.data['labels'] = os.listdir(self.data_fp) # list of file names to be indexed by output label
        rawX = []
        rawY = []
        class_samples = [] # number of samples in each class
        total_samples = 0 # total number of samples

        # DATA EXTRACTION
        print("    Extracting data from files...")
        for i in range(len(self.data['labels'])):
            root = os.path.join(self.data_fp, self.data['labels'][i])
            files = os.listdir(root) # get through data subdirectories
            class_samples.append(0)

            for f in files:
                x = np.genfromtxt(os.path.join(root, f), delimiter = ' ') # get sample from text file
                rawX.append(x) # raw sample input
                rawY.append(i) # raw sample output
                xx = keras.preprocessing.sequence.pad_sequences(x.T, maxlen = self.sequence_length, value = -1) # sequence samples     
                xx = np.reshape(xx, (1, xx.shape[1], xx.shape[0])) # reshape arrays to be stacked
                self.data['all']['x'] = np.vstack((self.data['all']['x'], xx)) # vertically stack samples

                class_samples[i] += 1
                total_samples += 1

        self.class_frequencies['Total Samples'] = total_samples
        for i in range(len(self.data['labels'])):            
            self.class_frequencies[self.data['labels'][i]] = {
                f'# {self.data["labels"][i]} Samples': class_samples[i],
                'Frequency': class_samples[i] / total_samples
            }

        y = np.array(rawY)
        n_cats = np.max(y) + 1
        y = keras.utils.to_categorical(y, n_cats) # one-hot encode outputs
        new_indices = np.arange(self.data['all']['x'].shape[0])
        np.random.shuffle(new_indices) # shuffle data
        self.data['all']['x'] = self.data['all']['x'][new_indices, :, :]
        self.data['all']['y'] = y[new_indices, :]
        self.data['raw'] = (rawX, rawY) # copy of raw data
        print("    Data extraction complete.")

        # DATA SPLITTING
        print("    Splitting data...")

        train_idx = round(split[0] * self.data['all']['x'].shape[0]) # train index
        val_idx = round(split[1] * self.data['all']['x'].shape[0]) # val/test index

        self.data['train']['x'] = self.data['all']['x'][:train_idx, :, :]
        self.data['train']['y'] = self.data['all']['y'][:train_idx, :]

        self.data['validation']['x'] = self.data['all']['x'][train_idx:(train_idx + val_idx), :, :]
        self.data['validation']['y'] = self.data['all']['y'][train_idx:(train_idx + val_idx), :]

        self.data['test']['x'] = self.data['all']['x'][(train_idx + val_idx):, :, :]
        self.data['test']['y'] = self.data['all']['y'][(train_idx + val_idx):, :]

        print("        Sequence Shapes:")
        print(f"            X Train: {self.data['train']['x'].shape}")
        print(f"            Y Train: {self.data['train']['y'].shape}")
        print(f"            X Validation: {self.data['validation']['x'].shape}")
        print(f"            Y Validation: {self.data['validation']['y'].shape}")
        print(f"            X Test: {self.data['test']['x'].shape}")
        print(f"            Y Test: {self.data['test']['y'].shape}")
        print("    Data splitting complete.")

        # DATA SCALING
        print("    Scaling data...")
        print("        Saving scalers...")
        skeys = ['x', 'y', 'z']
        for i in range(len(skeys)):
            # numpy forces us to scale each channel individually, thus
            # needing three total scalers
            self.scalers[skeys[i]].fit_transform(self.data['train']['x'][:, :, i])
            self.scalers[skeys[i]].transform(self.data['validation']['x'][:, :, i])
            self.scalers[skeys[i]].transform(self.data['test']['x'][:, :, i])
            sfn = f'./files/scalers/scaler-{skeys[i]}'
            joblib.dump(self.scalers[skeys[i]], sfn) # save scalers to disk
            print(f"            Saved scaler-{skeys[i]} to {sfn}")
        print("        Scalers saved.")
        print("    Data scaling complete.")
        print("    Saving processed data...")
        joblib.dump(self.data, './files/data/processed_data')
        print("    Processed data saved to ./files/data/processed_data")
        print("PREPROCESSING COMPLETE.")
    
    def cnn(self):
        model = models.Sequential([
            layers.Conv1D(128, 8, activation = 'relu', input_shape = self.data['all']['x'].shape[1:]),
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
            layers.Dense(len(self.data['labels']), activation = 'softmax')
        ])
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(), metrics = ['acc'])
        model.summary()
        return model
    
    def generate_models(self):
        print("GENERATING MODELS...")
        #self.MODELS['GRU'] = self.build_GRU_model()
        self.classifiers['cnn'] = self.cnn()
        print("MODEL GENERATION COMPLETE.")

    def evaluate_models(self, EPCHS = 400, BATCH = 16):
        print("EVALUATING MODELS...")
        print("    Training/testing models...")
        for k in self.classifiers:
            print(f"        Current Model: {k}")
            self.histories[k] = self.classifiers[k].fit(self.data['train']['x'], self.data['train']['y'], 
                                                        epochs = EPCHS, batch_size = BATCH, 
                                                        validation_data = (self.data['validation']['x'], self.data['validation']['y']), 
                                                        verbose = 2).history
            self.tests[k] = self.classifiers[k].evaluate(self.data['test']['x'], self.data['test']['y'], verbose = 1)
            print(f"            Model {k} Test Loss: {self.tests[k][0]}")
            print(f"            Model {k} Test Accuracy: {self.tests[k][1]}")
            print(f"        Model {k} evaluation complete.")
        print("    Training/testing complete.")
        print("    Generating and saving plots/metrics...")

        print("Label Dictionary")
        for i in range(len(self.data['labels'])):
                print(f"{i} : {self.data['labels'][i]}")
        
        for k in self.histories:
            loss_figure = plt.figure(figsize = (20, 15))
            al = loss_figure.add_subplot(111)

            acc_figure = plt.figure(figsize = (20, 15))
            aa = acc_figure.add_subplot(111)

            al.plot(range(EPCHS), self.histories[k]['loss'], label = f'{k} Training')
            al.plot(range(EPCHS), self.histories[k]['val_loss'], label = f'{k} Validation')
            al.plot(np.repeat(self.tests[k][0], EPCHS), label = f'{k} Testing')

            aa.plot(range(EPCHS), self.histories[k]['acc'], label = f'{k} Training')
            aa.plot(range(EPCHS), self.histories[k]['val_acc'], label = f'{k} Validation')
            aa.plot(np.repeat(self.tests[k][1], EPCHS), label = f'{k} Testing')
        
            al.title.set_text("Model Losses")
            al.set_xlabel("Epoch")
            al.set_ylabel("Loss")
            al.legend()

            aa.title.set_text("Model Accuracies")
            aa.set_xlabel("Epoch")
            aa.set_ylabel("Accuracy")
            aa.legend()

            preds = self.classifiers[k].predict(self.data['test']['x'])
            predy = np.argmax(preds, axis = 1)
            truey = np.argmax(self.data['test']['y'], axis = 1)
            labs = np.arange(len(self.data['labels']))[utils.multiclass.unique_labels(predy)]
            cm = metrics.confusion_matrix(truey, predy, labels = labs)
            cr = metrics.classification_report(truey, predy)

            conf_mat_fig = plt.figure(figsize = (20, 15))
            ac = conf_mat_fig.add_subplot(111)

            im = ac.matshow(cm)
            conf_mat_fig.colorbar(im)
            ac.set_xticklabels(labs)
            ac.set_yticklabels(labs)
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
    rec = HARProject()
    rec.execute()



