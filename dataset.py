# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:18:10 2019

@author: 12718
"""

import tensorflow as tf
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import Imageprocess

tf.reset_default_graph()

def writer_image():
    sub_dirs = [x[0] for x in os.walk('flower_photos', topdown = True)]
    sub_dirs.pop(0)
    file_name = sub_dirs[0]
    file_glob = os.path.join(file_name,'*.'+'jpg')
    file_list = glob.glob(file_glob)
    writer = tf.python_io.TFRecordWriter('daisy.tfrecords')
    with tf.Session() as sess:
        for file in file_list:
            image_raw = tf.gfile.FastGFile(file,'rb').read()
            image = tf.image.decode_jpeg(image_raw)
            image = tf.image.convert_image_dtype(image, dtype = tf.float32)
            image = tf.image.resize_images(image, [200,200], method = 1)
            image = image.eval()
#            print (image)
            
            label = 0
            height = 200
            width = 200
            channel = 3
            image = image.tostring()
    #        print (len(image))
            examples = tf.train.Example(features = tf.train.Features(feature = {
                    'image':tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                    'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
                    'height':tf.train.Feature(int64_list = tf.train.Int64List(value = [height])),
                    'width':tf.train.Feature(int64_list = tf.train.Int64List(value = [width])),
                    'channel':tf.train.Feature(int64_list = tf.train.Int64List(value = [channel]))}))
            writer.write(examples.SerializeToString())
        writer.close()
        
#def image_read_process():
reader = tf.TFRecordReader()
file_queue = tf.train.string_input_producer(['daisy.tfrecords'], num_epochs = 1)
_,example = reader.read(file_queue)
feature = tf.parse_single_example(example, features = {
        'image':tf.FixedLenFeature([], tf.string),
        'label':tf.FixedLenFeature([], tf.int64),
        'height':tf.FixedLenFeature([], tf.int64),
        'width':tf.FixedLenFeature([], tf.int64),
        'channel':tf.FixedLenFeature([], tf.int64)})

image_ = tf.decode_raw(feature['image'],tf.float32)
height,width,channel,label = feature['height'], feature['width'],feature['channel'],feature['label']
image_ = tf.reshape(image_,[200,200,3])   #这里不要用官方教程的set_shape，会报错

#### dataprocess
distorted_image = Imageprocess.slice_box(image_,200,200,None)
############batch
min_after_dequeue = 100
batch_size = 3
capacity = min_after_dequeue +3*batch_size
image_batch, label_batch = tf.train.shuffle_batch([distorted_image,label],
                                                  batch_size = batch_size,capacity = capacity,
                                                  min_after_dequeue = min_after_dequeue)
#    return image_batch, label_batch
'''然后直接把上面的batch代入训练函数即可'''
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord = coord)
    for i in range(2):
#        images,height_,width_,channel_ = sess.run([image_,height, width,channel])
#        images = np.reshape(images, [height_,width_,channel_])  #medhod1
#        image_batch, label_batch = image_read_process()
        print(image_batch.eval().shape,label_batch.eval())
#        plt.figure()
#        plt.imshow(images)
#        plt.show()
    coord.request_stop()
    coord.join(threads)
