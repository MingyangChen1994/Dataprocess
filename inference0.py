# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:47:55 2019

@author: 12718
"""

import tensorflow as tf

tf.reset_default_graph()

INPUT_SIZE = 784
IMAGE_SIZE = 28
HIDDEN_SIZE = 500
OUTPUT_SIZE = 10
NUM_CHANNEL = 1
CONV1_DEPTH = 8
CONV2_DEPTH = 16
FC_SIZE = 512

'''由于要加入regularization，所以在定义w时候，需要加入参数regular'''
def weights(shape, regular):
    w = tf.get_variable(name = 'weights', shape = shape,dtype = tf.float32,
                        initializer = tf.truncated_normal_initializer(stddev = 0.1))
    if regular is None:
        pass
    else:
        tf.add_to_collection('losses', regular(w))
    return w

def bias(shape):
    b = tf.get_variable(name = 'bias', shape = shape, dtype = tf.float32,
                        initializer = tf.constant_initializer(0.1))
    return b

'''for convolution layer'''
def weights_c(shape):
    w = tf.get_variable(name = 'weights_c', shape = shape,dtype = tf.float32,
                        initializer = tf.truncated_normal_initializer(stddev = 0.1))
    return w
#def inference(x, regular):
#    with tf.variable_scope('layer1'):
#        w = weights([INPUT_SIZE,HIDDEN_SIZE],regular)
#        b = bias([HIDDEN_SIZE])
#        layer1 = tf.nn.relu(tf.matmul(x,w)+b)
#    with tf.variable_scope('layer2'):
#        w = weights([HIDDEN_SIZE,OUTPUT_SIZE],regular)
#        b = bias([OUTPUT_SIZE])
#        layer2 = tf.matmul(layer1, w)+b
#    return layer2

'''convolution layer 只有全连接层的w需要正则化'''
def cov2d(x,w):
    cov2d = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME')
    return cov2d

def inference(x, regular, keep_prob):
    with tf.variable_scope('conv1'):
        w = weights_c([5,5,NUM_CHANNEL,CONV1_DEPTH])
        b = bias([CONV1_DEPTH])
        conv1 = tf.nn.relu(cov2d(x,w)+b)  #[batch_size, 28,28,8]
        pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')#[-1,14,14,8]
    with tf.variable_scope('conv2'):
        w = weights_c([5,5,CONV1_DEPTH,CONV2_DEPTH])
        b = bias([CONV2_DEPTH])
        conv2 = tf.nn.relu(cov2d(pool_1, w)+b) #[-1,14,14,16]
        pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') #[-1,7,16]
    with tf.variable_scope('FC1'):
        w = weights([7*7*16, FC_SIZE],regular)
        b = bias([FC_SIZE])
        pool_2_flat = tf.reshape(pool_2,[-1, 7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, w) + b)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.variable_scope('outlayer'):
        w = weights([FC_SIZE, OUTPUT_SIZE],regular)
        b = bias([OUTPUT_SIZE])
        h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop,w)+b)
    return h_fc2
        
        



