# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:16:41 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

'''随机调整图像的亮度，对比度，饱和度，色相，翻转等。随机顺序'''
def distorted_color(image, plan =0):
    if plan==0:
        image = tf.image.random_brightness(image, max_delta = 32./255.)
        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_saturation(image,lower = 0.5, upper = 1.5)
        image = tf.image.random_hue(image, max_delta = 0.2)
    elif plan==1:
        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_brightness(image, max_delta = 32./255.)
        image = tf.image.random_hue(image, max_delta = 0.2)
        image = tf.image.random_saturation(image,lower = 0.5, upper = 1.5)
    return tf.clip_by_value(image,0.,1.)  #调整色彩后可能会超出范围(0,1)

'''随机切原始图'''
def slice_box(image, height, width, bbox):
    if bbox is None:
        #bbox (batch_size, N, 4) N指的是N个bbox框
        bbox = tf.constant([0.,0.,1.,1.], dtype = tf.float32, shape = [1.,1.,4])
    if image.dtype!=tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)
    
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),bounding_boxes = bbox, min_object_covered = 0.4)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = tf.image.resize_images(distorted_image, [height, width], 
                                             method = np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distorted_color(distorted_image, plan = np.random.randint(2))
    return distorted_image

PATH = 'flower_photos'
sub_dirs = [x[0] for x in os.walk(PATH)]
sub_dirs.pop(0)
file_glob = os.path.join(sub_dirs[1],'*.'+'jpg')
file_list = glob.glob(file_glob)
file_name = file_list[0]
image_raw = tf.gfile.FastGFile(file_name,'rb').read()
image = tf.image.decode_jpeg(image_raw)
#image = tf.image.convert_image_dtype(image, dtype = tf.float32)
with tf.Session() as sess:
#    print(image.eval().shape)
#    plt.figure()
#    plt.imshow(image.eval())
#    plt.show()
    bbox = tf.constant([[[0.05, 0.05,0.9,0.7],[0.35,0.3,0.5,0.6]]])
   
    for i in range(6):
        image = slice_box(image, 200, 200, bbox)
        plt.figure()
        plt.imshow(image.eval())
        plt.show()
    




        
