# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:37:16 2018

@author: ashima.garg
"""

import config
import data
import model

if __name__ == "__main__":
    data = data.Data()
    data.read_train(config.TRAIN_X_PATH, config.TRAIN_Y_PATH)
    data.preprocess()
    data.split()
    print("data read")
    
    model = model.Model()
    model.build()
    print("model build")
    
    model.train(data)
    print("model trained")
    
    model.test(data)
    print("model tested")
    '''
    data.read_test(config.TEST_X_PATH)
    data.preprocess()
    print("model predicted")
    '''
    
