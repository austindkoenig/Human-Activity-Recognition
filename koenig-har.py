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

class HARSystem(object):
    def __init__(self, data_fp = './data/'):
        self.data_fp = data_fp
        self.raw_data = None
        self.data = None
        self.train = None
        self.val = None
        self.test = None
    
    def preprocess(self):
        label_bank = os.listdir(self.data_fp)
        for i in range(len(label_bank)):
            root = os.path.join(self.data_fp, label_bank[i])
            files = os.listdir(root)
            for f in files:
                x = np.genfromtxt(os.path.join(root, f), delimiter = ' ')
                y = label_bank[i]

    def build_model(self):
        pass

    def evaluate_model(self):
        pass



if __name__ == "__main__":
    rec = HARSystem()
    rec.preprocess()



