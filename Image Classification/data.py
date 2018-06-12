# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:36:46 2018

@author: ashima.garg
"""

import numpy as np
import pandas as pd
import config
import pickle 
from sklearn.cross_validation import train_test_split

class DATA:
    def __init__(self):
        self.batch_size = config.BATCH_SIZE
        self.dataX = None
        self.dataY = None
        self.size = None
        self.data_index = 0
        self.dataX_test = None
        self.dataY_test = None
        
    def read_train(self, filenameX, filenameY):
        file = open(filenameX, 'rb')
        self.dataX = pickle.load(file)
        file.close()
        file = open(filenameY, 'rb')
        self.dataY = pickle.load(file)
        file.close()
        
    def read_test(self, filenameX):
        file = open(filenameX, 'rb')
        self.dataX = pickle.load(file)
        file.close()
        
    def preprocess(self):
        self.dataX = self.dataX/255.
        
    def split(self):
        self.dataX, self.dataX_test, self.dataY, self.dataY_test = train_test_split(self.dataX, self.dataY, test_size = 0.22, random_state = 0)
        
    def generate_batch(self):
        batch_X = self.dataX[self.data_index:self.data_index + self.batch_size]
        batch_Y = self.dataY[self.data_index:self.data_index + self.batch_size]
        self.data_index = (self.data_index + self.batch_size) % self.size
        return batch_X, batch_Y
    