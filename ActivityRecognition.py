'''
Activity Recognition
Austin Koenig
2019

Data source: https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer
    Description: Each datum is a variable-length sequence of triaxial accelerometer data that
    corresponds to a particular activity from the following list:
        - Brush_teeth      - Climb_stairs
        - Comb_hair        - Descend_stairs    
        - Drink_glass      - Eat_meat    
        - Eat_soup         - Getup_bed    
        - Liedown_bed      - Pour_water    
        - Sitdown_chair    - Standup_chair    
        - Use_telephone    - Walk
'''

# Usual imports+
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil # to delete directories
import joblib # used to pickle data

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers

from sklearn import preprocessing
from sklearn import metrics
from sklearn import utils

np.random.seed(12)


checkpoints = {
    'extract': False,
    'preprocess': False,
    'model_gen': False,
    'model_eval': False
}

dirs = {
    'root': './files/',
    'data': './files/data/',
    'figures': './files/figures/',
    'scalers': './files/scalers/',
    'models': './files/models/',
    'metrics': './files/metrics/'
}

data = { 
    'seqlen': 1024,
    'raw': {
        'x': [],
        'y': []
    }
}

scalers = {}
classifiers = {}
histories = {}
tests = {}


# Main functions

def extract_data_from_dir(filepath = './data/'):
    _create_dirs()
    data['labs'] = os.listdir(filepath)
    for i in range(len(data['labs'])):
        root = os.path.join(filepath, data['labs'][i])
        files = os.listdir(root) # go through data directories
        #class_samples.append(0)

        for f in files:
            x = np.genfromtxt(os.path.join(root, f), delimiter = ' ') # get sample from text file

            data['raw']['x'].append(x) # raw sample input
            data['raw']['y'].append(i) # raw sample output

            xx = keras.preprocessing.sequence.pad_sequences(x.T, maxlen = data['seqlen'], value = -1) # short-sequence samples                
            xx = np.reshape(xx, (1, xx.shape[1], xx.shape[0])) # reshape arrays to be stacked along axis 0
            data['all']['x'] = np.vstack((data['all']['x'], xx)) # vertically stack samples

            #class_samples[i] += 1
            #total_samples += 1

    y = np.array(data['raw']['y'])
    n_cats = np.max(y) + 1
    y = keras.utils.to_categorical(y, n_cats) # one-hot encode outputs
    new_indices = np.arange(data['all']['x'].shape[0])
    np.random.shuffle(new_indices) # shuffle data
    data['all']['x'] = data['all']['x'][new_indices, :, :]
    data['all']['y'] = y[new_indices, :]
    joblib.dump(data, os.path.join(dirs['data'], 'raw_data'))
    checkpoints['extract'] = True

def preprocess_data(train_split = 0.8):
    if not checkpoints['extract']:
        extract_data_from_dir()

    data = joblib.load(os.path.join(dirs['data'], 'raw_data'))

    # data splitting
    train_idx = round(train_split * data['all']['x'].shape[0])

    data['trn']['x'] = data['all']['x'][:train_idx, :, :]
    data['trn']['y'] = data['all']['y'][:train_idx, :]
    data['tst']['x'] = data['all']['x'][train_idx:, :, :]
    data['tst']['y'] = data['all']['y'][train_idx:, :]

    # data scaling
    keys = ['x', 'y', 'z']
    for k in range(len(keys)):
        scalers[keys[k]].fit_transform(data['trn']['x'][:, :, k])
        scalers[keys[k]].transform(data['tst']['x'][:, :, k])
        joblib.dump(scalers[keys[k]], os.path.join(dirs['scalers'], f'scaler-{keys[k]}'))
    joblib.dump(data, os.path.join(dirs['data'], 'processed_data'))
    checkpoints['preprocess'] = True

def generate_models():
    classifiers['cnn'] = cnn()
    for key in classifiers:
        joblib.dump(classifiers[key], os.path.join(dirs['models'], key))
    checkpoints['model_gen'] = True

def cnn():
    model = models.Sequential([
        layers.Conv1D(128, 8, activation = 'relu', input_shape = data['all']['x'].shape[1:]),
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
        layers.Dense(len(data['labs']), activation = 'softmax')
    ])
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(), metrics = ['acc'])
    model.summary()
    return model

def evaluation(EPCHS = 200, BATCH = 16):
    for k in classifiers:
        print(f"Classifier: {k}")
        histories[k] = classifiers[k].fit(data['trn']['x'], data['trn']['y'], 
                                            epochs = EPCHS, batch_size = BATCH, 
                                            validation__split = 0.2, 
                                            verbose = 2).history
        tests[k] = classifiers[k].evaluate(self.test['x'], self.test['y'], verbose = 1)
        print(f"    Model {k} Test Loss: {self.TESTS[k][0]}")
        print(f"    Model {k} Test Accuracy: {self.TESTS[k][1]}")
        joblib.dump(classifiers[k], os.path.join(dirs['models'], k + '_trained'))

# Helper functions

def _create_dirs():
    _clean_dirs()
    for d in dirs:
        if not os.path.exists(dirs[d]):
            os.makedirs(dirs[d])
    _prepare_dicts()

def _prepare_dicts():
    data_keys = ['all', 'trn', 'val', 'tvl', 'tst']
    scaler_keys = ['x', 'y', 'z']

    for k in data_keys:
        data[k] = {
            'x': np.empty((0, data['seqlen'], 3)),
            'y': None
        }
    
    for k in scaler_keys:
        scalers[k] = preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False)

def _clean_dirs():
    if os.path.exists('./files/'):
        shutil.rmtree('./files/')


