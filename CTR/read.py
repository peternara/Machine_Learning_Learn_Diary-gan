# coding=utf-8
import tensorflow as tf
import numpy as np
import os

'''
读取CTR数据
'''


class CTRReader(object):
    def __init__(self, path, pattem, batch_size=0, num_classes=None, is_training=True):
        self.num_classes = num_classes
        files = tf.gfile.Glob(os.path.join(path, pattem))  # 遍历所有文件
        self.reader = tf.TextLineReader()
        self.is_training = is_training
        if is_training:
            self.file_queue = tf.train.string_input_producer(files)  # 构造文件队列
            self.batch_size = batch_size
        else:
            self.file_queue = tf.train.string_input_producer(files, shuffle=False)
            test_file = tf.gfile.FastGFile(os.path.join(path, pattem), 'rb')
            line_count = 0
            for _ in test_file.readlines():
                line_count += 1
            print "测试集总数: {}".format(line_count)
            self.batch_size = line_count

    def read(self):
        file_name, file_content = self.reader.read(self.file_queue)
        data, label = self.parse_csv(file_content)
        if self.is_training:
            datas, labels = tf.train.shuffle_batch([data, label],
                                                   batch_size=self.batch_size,
                                                   capacity=1000 + 3 * self.batch_size,
                                                   min_after_dequeue=1000,
                                                   num_threads=4)
            return datas, labels
        else:
            datas, labels = tf.train.batch([data, label],
                                           batch_size=self.batch_size,
                                           num_threads=4)
            return datas, labels

    def parse_csv(self, content):
        data = tf.decode_csv(content, record_defaults=np.zeros([20, 1], dtype=np.float32).tolist())  # 解析CSV
        # 从中提取需要的字段
        data_part_1 = tf.slice(data, [3], [6])
        data_part_2 = tf.slice(data, [11], [8])
        data_part = tf.concat([[data_part_1], [data_part_2]], 1)[0]
        data_label = data[-1]
        data_label = tf.cast(data_label, tf.uint8)
        data_label = tf.one_hot(data_label, self.num_classes)
        return data_part, data_label


if __name__ == '__main__':
    # 测试读取
    sess = tf.InteractiveSession()
    reader = CTRReader('data', 'train.csv', 5, 2)
    datas, labels = reader.read()
    tf.train.start_queue_runners(sess=sess)
    print datas.eval()
    print labels.eval()
