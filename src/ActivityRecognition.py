'''
Human Activity Recognizer
Created by Austin Koenig
Version: 0.0.1

The object of this project is to create a classifier which can differentiate between 14 different human activities from data collected via a triaxial accelerometer mounted to the subjects' wrists.

Accelerometer Specifications
----------------------------
    Type : tri-axial accelerometer
    Measurement range : [- 1.5g; + 1.5g]
    Sensitivity : 6 bits per axis
    Output data rate : 32 Hz

Data Source: https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer

Note that files with a `.jl` extension are serialized joblib files.

See `README.md` for more details.
'''
# usual imports
import joblib
import json
import os
import shutil
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12) # seed for reproducibility

# keras imports
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras import callbacks

# sklearn imports
from sklearn import preprocessing
from sklearn import metrics
from sklearn import utils


class TimingCallback(callbacks.Callback):
    '''
    This class comes from solution at: https://github.com/keras-team/keras/issues/5105
    '''
    def __init__(self):
        self.logs = []
    
    def on_epoch_begin(self, epoch, logs = {}):
        self.starttime = time.time()
    
    def on_epoch_end(self, epoch, logs = {}):
        self.logs.append(time.time() - self.starttime)


class ActivityRecognizer(object):
    def __init__(self):
        pass
    
    def preprocess_data(self, input_filepath = './data/', output_filepath = './output/', split = 0.85, sequence_length = 512):
        '''
        Extract and preprocess data from database or other type of storage system and restore for later use.
        '''
        self._reset_log() # reset the log file

        metadata = {}
        scalers = {}
        data = {}

        metadata['input_filepath'] = input_filepath
        metadata['output_filepath'] = output_filepath
        metadata['train_test_split'] = split
        metadata['sequence_length'] = sequence_length
        metadata['labels'] = os.listdir(metadata['input_filepath'])
        metadata['num_categories'] = len(metadata['labels'])
        metadata['class_samples'] = {}
        metadata['total_samples'] = 0
        metadata['class_frequencies'] = {}
        metadata['shapes'] = None

        scalers['x'] = preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False)
        scalers['y'] = preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False)
        scalers['z'] = preprocessing.MinMaxScaler(feature_range = (0, 1), copy = False)

        data['all'] = {
            'x': np.empty((0, metadata['sequence_length'], 3)),
            'y': np.array([])
        }

        data['train'] = {
            'x': np.empty((0, metadata['sequence_length'], 3)),
            'y': np.array([])
        }

        data['test'] = {
            'x': np.empty((0, metadata['sequence_length'], 3)),
            'y': np.array([])
        }

        for i in range(len(metadata['labels'])):
            root = os.path.join(metadata['input_filepath'], metadata['labels'][i])
            files = os.listdir(root) # go through data subdirectories
            metadata['class_samples'][metadata['labels'][i]] = 0

            for f in files:
                x = np.genfromtxt(os.path.join(root, f), delimiter = ' ') # get sample from text file
                x = keras.preprocessing.sequence.pad_sequences(x.T, maxlen = metadata['sequence_length'], value = 0)
                x = np.reshape(x, (1, x.shape[1], x.shape[0]))
                data['all']['x'] = np.vstack((data['all']['x'], x))
                data['all']['y'] = np.append(data['all']['y'], i)
                metadata['class_samples'][metadata['labels'][i]] += 1
                metadata['total_samples'] += 1
        
        for i in range(len(metadata['labels'])):
            metadata['class_frequencies'][metadata['labels'][i]] = round(metadata['class_samples'][metadata['labels'][i]] / metadata['total_samples'], 6)
        
        data['all']['y'] = keras.utils.to_categorical(data['all']['y'], metadata['num_categories']) # one hot encode outcome

        # shuffle data
        new_indices = np.arange(data['all']['x'].shape[0])
        np.random.shuffle(new_indices)
        data['all']['x'] = data['all']['x'][new_indices, :, :]
        data['all']['y'] = data['all']['y'][new_indices, :]

        metadata['shapes'] = {
            'x': data['all']['x'].shape,
            'y': data['all']['y'].shape
        }
        
        split_index = round(split * data['all']['x'].shape[0])

        data['train']['x'] = data['all']['x'][:split_index, :, :]
        data['train']['y'] = data['all']['y'][:split_index, :]

        data['test']['x'] = data['all']['x'][split_index:, :, :]
        data['test']['y'] = data['all']['y'][split_index:, :]

        print("## Preprocessing\n")

        shapes = {
            'train_split': split,
            'test_split': round(1 - split, 4),
            'shapes': {
                'x_train': data['train']['x'].shape,
                'y_train': data['train']['y'].shape,
                'x_test': data['test']['x'].shape,
                'y_test': data['test']['y'].shape,
            }
        }

        print("```")
        print(json.dumps(shapes, indent = 4))
        print("```")

        axis_keys = ['x', 'y', 'z']
        for i in range(len(axis_keys)):
            scalers[axis_keys[i]].fit_transform(data['train']['x'][:, :, i])
            scalers[axis_keys[i]].transform(data['test']['x'][:, :, i])
        
        json.dump(metadata, open('./output/metadata.json', 'w'), indent = 4)
        joblib.dump(scalers, './output/scalers.joblib')
        joblib.dump(data['all'], './output/data.joblib')
        joblib.dump(data['train'], './output/train.joblib')
        joblib.dump(data['test'], './output/test.joblib')
    
    def generate_model(self):
        '''
        Generate model and store for later use.
        '''
        metadata = json.load(open('./output/metadata.json', 'r'))
        model = models.Sequential([
            layers.Conv1D(512, 4, activation = 'relu', input_shape = metadata['shapes']['x'][1:]),
            layers.Conv1D(512, 4, activation = 'relu'),
            layers.Conv1D(512, 4, activation = 'relu'),
            layers.MaxPooling1D(8),
            layers.Dropout(0.5),
            layers.Conv1D(512, 4, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Conv1D(512, 4, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Conv1D(512, 4, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.MaxPooling1D(6),
            layers.Dropout(0.5),
            layers.Conv1D(512, 2, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Conv1D(512, 2, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Conv1D(512, 2, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.MaxPooling1D(4),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(1024, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Dense(1024, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Dense(1024, activation = 'relu', kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(metadata['num_categories'], activation = 'softmax')
        ])
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(learning_rate = 0.01, nesterov = True), metrics = ['acc'])
        print("## Model Hierarchy")
        print("```")
        model.summary()
        print("```")
        model.save('./output/model.hdf5')
    
    def train_model(self, num_epochs = 256, batch = 4, val_split = 0.2):
        '''
        Train a model from the saved preprocessed data and saved model and restore model for later use.
        '''
        print("## Model Training\n")
        model = models.load_model('./output/model.hdf5')
        train = joblib.load('./output/train.joblib')
        metadata = json.load(open('./output/metadata.json', 'r'))

        metadata['epochs'] = num_epochs
        metadata['batch_size'] = batch
        metadata['train_val_split'] = val_split

        savebest = callbacks.ModelCheckpoint('./output/model.hdf5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
        timing = TimingCallback()

        print("```")
        history = model.fit(train['x'], train['y'], 
                            epochs = num_epochs,
                            batch_size = batch,
                            verbose = 2,
                            validation_split = val_split,
                            callbacks = [savebest, timing]).history
        print("```")

        metadata['train_time'] = f"{round(sum(timing.logs), 4)} s"
        metadata['mean_epoch_train_time'] = f"{round(sum(timing.logs) / metadata['epochs'], 4)} s"

        joblib.dump(history, './output/history.joblib')
        json.dump(metadata, open('./output/metadata.json', 'w'), indent = 4)
        #model.save('./output/model.hdf5')
    
    def evaluate_model(self):
        '''
        Evaluate stored model and generate output files:
            - Log file
                - Confusion Matrix
                - Classification report
                    - Accuracy
                    - Precision
                    - Recall
                    - F1
            - Accuracy plot
            - Loss plot
            - Heatmap of confusion matrix
        '''
        model = models.load_model('./output/model.hdf5')
        test = joblib.load('./output/test.joblib')
        history = joblib.load('./output/history.joblib')
        metadata = json.load(open('./output/metadata.json', 'r'))

        evaluation = model.evaluate(test['x'], test['y'], verbose = 2)

        print("## Model Evaluation\n")

        preds = model.predict(test['x'])
        predy = np.argmax(preds, axis = 1)
        truey = np.argmax(test['y'], axis = 1)
        labs = np.arange(len(metadata['labels']))[utils.multiclass.unique_labels(predy)]

        cm = metrics.confusion_matrix(truey, predy, labels = labs)
        cr = metrics.classification_report(truey, predy)

        print("### Class Dictionary\n")

        print("```")
        for i in range(metadata['num_categories']):
            print(f"{i} : {metadata['labels'][i]}")
        print("```")

        results = {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'confusion_matrix': cm,
            'classification_report': cr
        }

        print("### Results\n")

        print("```")
        print(f"Test Loss: {round(results['loss'], 4)}")
        print(f"Test Accuracy: {round(results['accuracy'], 4)}")
        print(f"\nConfusion Matrix: \n{results['confusion_matrix']}")
        print(f"\nClassification Report: \n{results['classification_report']}")
        print("```")

        print('### Metadata\n')
        print(json.dumps(metadata, indent = 4))

        loss_figure = plt.figure(figsize = (20, 15))
        loss_axis = loss_figure.add_subplot(111)
        loss_axis.plot(range(metadata['epochs']), history['loss'], label = 'Train')
        loss_axis.plot(range(metadata['epochs']), history['val_loss'], label = 'Validation')
        loss_axis.plot(np.repeat(results['loss'], metadata['epochs']), label = 'Best Test')
        loss_axis.title.set_text("Losses")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Loss")
        loss_axis.legend()

        acc_figure = plt.figure(figsize = (20, 15))
        acc_axis = acc_figure.add_subplot(111)
        acc_axis.plot(range(metadata['epochs']), history['acc'], label = 'Train')
        acc_axis.plot(range(metadata['epochs']), history['val_acc'], label = 'Validation')
        acc_axis.plot(np.repeat(results['accuracy'], metadata['epochs']), label = 'Best Test')
        acc_axis.title.set_text("Losses")
        acc_axis.set_xlabel("Epoch")
        acc_axis.set_ylabel("Loss")
        acc_axis.legend()

        cm_figure = plt.figure(figsize = (20, 15))
        cm_axis = cm_figure.add_subplot(111)
        im = cm_axis.matshow(cm)
        cm_figure.colorbar(im)
        cm_axis.set_xticklabels(labs)
        cm_axis.set_yticklabels(labs)
        cm_axis.set_xlabel("Predicted")
        cm_axis.set_ylabel("True")

        joblib.dump(results, './output/results.joblib')
        loss_figure.savefig('./output/loss.pdf')
        acc_figure.savefig('./output/acc.pdf')
        cm_figure.savefig('./output/cm.pdf')

    def _reset_log(self):
        self._sanitize_directories()
        sys.stdout = open('./output/log.md', 'w') # log file
        print("# <center>Activity Recognizer Output Log</center>\n")

    def _sanitize_directories(self):
        if os.path.exists('./output/'): # if the output directory exists
            shutil.rmtree('./output/') # delete it
        os.makedirs('./output/') # make a new output directory