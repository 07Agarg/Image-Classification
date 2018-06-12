# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:37:07 2018

@author: ashima.garg
"""
import os
import config
import tensorflow as tf
from neural_network import Conv_Layer, FC_Layer, Outer_Layer
import numpy as np

class Model:
    def __init__(self):
        self.x_holder= tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])
        self.y_holder = tf.placeholder(tf.float32, shape=[None, config.NUM_CLASSES])
        self.loss = None
        self.output = None
        
    def build(self):
        
        
    def train(self):
        
        
    def test(self):
        


