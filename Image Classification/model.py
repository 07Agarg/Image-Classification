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
        self.x_holder= tf.placeholder(tf.float32, shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 1])
        self.y_holder = tf.placeholder(tf.float32, shape=[None, config.NUM_CLASSES])
        self.loss = None
        self.output = None
        
    def build_network(self):
        input_ = self.x_holder
        conv1 = Conv_Layer([7, 7, 1, 32])
        input_ = conv1.feed_forward(input_)
        conv2 = Conv_Layer([5, 5, 32, 64])
        input_ = conv2.feed_forward(input_)
        conv3 = Conv_Layer([3, 3, 64, 128])
        input_ = conv3.feed_forward(input_)
        input_shape = input_.get_shape().as_list()
        input_ = tf.reshape(input_, [config.BATCH_SIZE, input_shape[1] * input_shape[2] * input_shape[3]])
        dim = input_.get_shape()[1].value
        fc1 = FC_Layer([dim, 1000])
        input_ = fc1.feed_forward(input_)
        fc2 = FC_Layer([1000, 500])
        input_ = fc2.feed_forward(input_)
        out1 = Outer_Layer([500, 6])
        self.output = out1.feed_forward(input_)
        self.loss = self.loss_func()
        
    def train(self, data):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session as session:
            session.run(tf.global_variables_initializer())
            print('All variables initialized')
            total_batch = int(len(data.dataX)/config.BATCH_SIZE)
            for epoch in range(config.NUM_EPOCHS):
                cost = 0
                for batch in range(total_batch):
                    batchX, batchY = data.generate_batch()
                    feed_dict = {self.x_holder: batchX, self.y_holder: batchY}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict = feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    cost += loss_val / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost))
                
            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            print("Model saved in path: %s" % save_path)
        
    def test(self, data):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            for i in range(len(data.dataX_test)):
                feed_dict = {self.x_holder: [data.dataX_test[i]]}
                predicted = np.rint(session.run(self.output, feed_dict=feed_dict))
                print('Actual:', data.dataY_test[i], 'Predicted:', predicted)
        
    def evaluate(self, data):
        f= open(os.path.join(config.RESULT_DIR, "/outputs"), 'w')
        with tf.Session as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            for i in range(len(data.dataX)):
                feed_dict = {self.x_holder: [data.dataX[i]]}
                predicted = np.rint(session.run(self.output, feed_dict = feed_dict))
                f.write(predicted)
                #print('Predicted:', predicted)
        f.close()
        
    def loss_func(self):
        return tf.reduce_mean(-tf.reduce_sum(self.y_holder * tf.log(self.output), axis=0))