# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:36:23 2018

@author: ashima.garg
"""

import os

ROOT_DIR = os.path.dirname(os.apth.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL')

TRAIN_X_PATH = "train\\train_x.obj"
TRAIN_Y_PATH = "train\\labels_.obj"
TEST_X_PATH = "test\\test_x.obj"

NUM_EPOCHS = 500

LEARNING_RATE = 0.01

NUM_CLASSES = 6
IMAGE_SIZE = 108

BATCH_SIZE = 128

SEED = 120