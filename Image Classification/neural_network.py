# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:36:56 2018

@author: ashima.garg
"""
import tensorflow as tf

class Conv_layer:
    def __init__(self, shape):
        self.weight = tf.Variable(tf.random_normal(shape=shape ,stddev=0.01))
        self.bias = tf.Variable(tf.constant(0.1, shape = [shape[1]]))
    
    def feed_forward(self, input_):
        conv = tf.nn.conv2d(input_, self.weight, [1, 2, 2, 1], padding="SAME")
        return tf.nn.relu(tf.nn.bias_add(conv, self.bias))
        
        
class FC_Layer:
    def __init__(self, shape):
        self.weight = tf.Variable(tf.random_normal(shape=shape ,stddev=0.01))
        self.bias = tf.Variable(tf.constant(0.1, shape = [shape[1]]))
        
    def feed_forward(self, input_):
        output_ = tf.nn.relu(tf.matmul(input_, self.weight) + self.bias)
        return output_
        
class Outer_Layer:
    def __init__(self, shape):
        self.weight = tf.Variable(tf.random_normal(shape=shape ,stddev=0.01))
        self.bias = tf.Variable(tf.constant(0.1, shape = [shape[1]]))
        
    def feed_forward(self, input_):
        output_ = tf.nn.softmax(tf.matmul(input_, self.weight) + self.bias)
        return output_
        
