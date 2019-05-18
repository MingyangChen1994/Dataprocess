# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:17:54 2019

@author: 12718
"""

import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
#模拟海量数据情况下写入不容的文件
num_shards = 2  #共写入两个文件
instance_per_shard = 2 #每个文件写入多少数据
for i in range(num_shards):
    filename = ('data.tfrecords-%.5d-of-%.5d'%(i,num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instance_per_shard):
        example = tf.train.Example(features = tf.train.Features(feature = {
                'i':_int64_feature(i),
                'j':_int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
    
##读取
files = tf.train.match_filenames_once('data.tfrecords-*')
#tf.train.string_input_producer函数创建输入队列
filename_queue = tf.train.string_input_producer(files,shuffle = True, num_epochs = None)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features = {
        'i':tf.FixedLenFeature([],tf.int64),
        'j':tf.FixedLenFeature([],tf.int64)})
#with tf.Session() as sess:
#    ###tf.train.match_filenames_once需要初始化
#    sess.run(tf.local_variables_initializer())
#    print (sess.run(files))
#    
#    ##声明线程
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
#    for i in range(9):
#        print (sess.run([features['i'], features['j']]))
#    coord.request_stop()
#    coord.join(threads)
    
###组合训练数据
examples, label = features['i'], features['j']
batch_size = 3
###组合样例队列中最多可以存储的样例个数。队列太大，占内存，太小，batch不好出
capacity = 1000+3*batch_size

#example_batch, label_batch = tf.train.batch([examples,label], batch_size = batch_size,
#                                            capacity = capacity)

##### tf.train.shuffle_bach
example_batch, label_batch = tf.train.shuffle_batch([examples, label],batch_size = batch_size,
                                                    capacity = capacity, min_after_dequeue = 30)
## min_after_dequeue限制了出队时队列中剩下的最少的元素个数保证随机有效性
with tf.Session() as sess:
    tf.local_variables_initializer().run()   #这里需要使用局部变量,因为使用了start_queue_runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    for i in range(5):
        cur_example_batch,cur_label_batch = sess.run([example_batch, label_batch])
        print (cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)
    


###############################################################################
def next_batch():
    datasets = np.arange(20)
    input_queue = tf.train.slice_input_producer([datasets],num_epochs = 1)
    data_batches = tf.train.batch(input_queue,batch_size = 5, capacity = 20)
    return data_batches

if __name__ == '__main__':
    data_batches = next_batch()
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    try:
        while not coord.should_stop():
            print (sess.run(data_batches))
    except tf.errors.OutOfRangeError:
        print('complete')
    finally:
        coord.request_stop()
    coord.join(threads)










