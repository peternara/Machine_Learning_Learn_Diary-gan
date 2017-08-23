import tensorflow as tf
import numpy as np
import os


class MnistReader(object):
    def __init__(self,file_path, is_training=True):
    
        self.filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(file_path))
        #根据文件名生成一个队列
        self.is_training = is_training
        if is_training:
            self.batch_size = 55000
        else:
            self.batch_size = 5
  
            
    def read_image(self,file_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)#返回文件名和文件
        features = tf.parse_single_example(
            serialized_example,
            features={
              'image_raw': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64),
              })#将文件解析为张量
              
        #将字符串解析成图像对应的像素数组
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([784])
        #归一化
        image = tf.cast(image, tf.float32) * (1. / 255)
        label = tf.cast(features['label'], tf.int32)
        return image, label

    def read_image_batch(self,filename_queue, batch_size):
        img, label = self.read_image(filename_queue)
        capacity = 1000 + 3 * batch_size
        #获得batch_size个图像
        image_batch = tf.train.batch([img], batch_size=batch_size, capacity=capacity, num_threads=10)
        #one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))
        return image_batch
        
        
    def read(self):
        train_images = self.read_image_batch(self.filename_queue, self.batch_size)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            i = 0
            #启动队列
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            try:
                while not coord.should_stop() and i<1:
                    img_ = sess.run(train_images)
                    i+=1
            except tf.errors.OutOfRangeError:
                print("Done:")
            finally:
                coord.request_stop()
            coord.join(threads)

        return img_

  