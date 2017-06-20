import tensorflow as tf
import os

class TFRecord_Generator(object):
    def __init__(self,
                 train_path='train/*.jpg',
                 test_path='test/*.jpg',
                 crop_resize=True,
                 standarlizing=True,

                 ):
        pass