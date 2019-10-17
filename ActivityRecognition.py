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
import sys
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
sys.stdout = open('output_log.txt', 'w')


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

            xx = keras.preprocessing.sequence.pad_sequences(x.T, maxlen = data['seqlen'], value = -1)             
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

def preprocess_data(splits = (0.7, 0.15)):
    if not checkpoints['extract']:
        extract_data_from_dir()

    data = joblib.load(os.path.join(dirs['data'], 'raw_data'))

    # data splitting
    train_idx = round(splits[0] * data['all']['x'].shape[0])
    val_idx = round(splits[1] * data['all']['x'].shape[0])

    data['trn']['x'] = data['all']['x'][:train_idx, :, :]
    data['trn']['y'] = data['all']['y'][:train_idx, :]
    data['val']['x'] = data['all']['x'][train_idx:(train_idx + val_idx), :, :]
    data['val']['y'] = data['all']['y'][train_idx:(train_idx + val_idx), :]
    data['tvl']['x'] = data['all']['x'][:(train_idx + val_idx), :, :]
    data['tvl']['y'] = data['all']['y'][:(train_idx + val_idx), :]
    data['tst']['x'] = data['all']['x'][(train_idx + val_idx):, :, :]
    data['tst']['y'] = data['all']['y'][(train_idx + val_idx):, :]

    # data scaling
    keys = ['x', 'y', 'z']
    for k in range(len(keys)):
        scalers[keys[k]].fit_transform(data['trn']['x'][:, :, k])
        scalers[keys[k]].transform(data['val']['x'][:, :, k])
        scalers[keys[k]].transform(data['tvl']['x'][:, :, k])
        scalers[keys[k]].transform(data['tst']['x'][:, :, k])
        joblib.dump(scalers[keys[k]], os.path.join(dirs['scalers'], f'scaler-{keys[k]}'))
    joblib.dump(data, os.path.join(dirs['data'], 'processed_data'))
    checkpoints['preprocess'] = True

def generate_models():
    if not checkpoints['preprocess']:
        preprocess_data()
    classifiers['cnn'] = cnn()
    for key in classifiers:
        classifiers[key].save(os.path.join(dirs['models'], key + '.h5'))
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

def evaluation(EPCHS = 10, BATCH = 16):
    if not checkpoints['model_gen']:
        generate_models()

    for k in classifiers:
        print(f"Classifier: {k}")
        classifiers[k] = models.load_model(os.path.join(dirs['models'], k + '.h5'))
        histories[k] = classifiers[k].fit(data['trn']['x'], data['trn']['y'], 
                                          epochs = EPCHS, batch_size = BATCH, 
                                          validation_data = (data['val']['x'], data['val']['y']), verbose = 2).history
        tests[k] = classifiers[k].evaluate(data['tst']['x'], data['tst']['y'], verbose = 1)

        print(f"    Model {k} Test Loss: {tests[k][0]}")
        print(f"    Model {k} Test Accuracy: {tests[k][1]}")

        loss_fig = plt.figure(figsize = (20, 15))
        al = loss_fig.add_subplot(111)
        al.plot(range(EPCHS), histories[k]['loss'], label = f'{k} Training')
        al.plot(range(EPCHS), histories[k]['val_loss'], label = f'{k} Validation')
        al.plot(np.repeat(tests[k][0], EPCHS), label = f'{k} Testing')
        al.title.set_text("Model Losses")
        al.set_xlabel("Epoch")
        al.set_ylabel("Loss")
        al.legend()

        acc_fig = plt.figure(figsize = (20, 15))
        aa = acc_fig.add_subplot(111)
        aa.plot(range(EPCHS), histories[k]['acc'], label = f'{k} Training')
        aa.plot(range(EPCHS), histories[k]['val_acc'], label = f'{k} Validation')
        aa.plot(np.repeat(tests[k][1], EPCHS), label = f'{k} Testing')
        aa.title.set_text("Model Accuracies")
        aa.set_xlabel("Epoch")
        aa.set_ylabel("Accuracy")
        aa.legend()

        preds = classifiers[k].predict(data['tst']['x'])
        predy = np.argmax(preds, axis = 1)
        truey = np.argmax(data['tst']['y'], axis = 1)
        labs = np.arange(len(data['labs']))[utils.multiclass.unique_labels(predy)]

        cm = metrics.confusion_matrix(truey, predy, labels = data['labs'][labs]) # confusion matrix
        cr = metrics.classification_report(truey, predy) # classification report

        conf_mat_fig = plt.figure(figsize = (20, 15))
        ac = conf_mat_fig.add_subplot(111)
        im = ac.matshow(cm)
        conf_mat_fig.colorbar(im)
        ac.set_xticklabels(data['labs'][labs])
        ac.set_yticklabels(data['labs'][labs])
        ac.set_xlabel('Predicted')
        ac.set_ylabel('True')

        # save stuff
        conf_mat_fig.savefig(os.path.join(dirs['figures'], f'{k}-confusion-matrix.pdf'))
        joblib.dump(cr, os.path.join(dirs['figures'], f'{k}-classification-report'))
        joblib.dump(cm, os.path.join(dirs['figures'], f'{k}-confusion-matrix'))
        classifiers[k].save(os.path.join(dirs['models'], k + '-trained.h5'))
        joblib.dump(histories[k], os.path.join(dirs['metrics'], f'{k}-history'))
        joblib.dump(tests[k], os.path.join(dirs['metrics'], f'{k}-test'))
    checkpoints['model_eval'] = True

def execute():
    extract_data_from_dir()
    preprocess_data()
    generate_models()
    evaluation()

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


