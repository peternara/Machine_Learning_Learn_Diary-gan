# coding=utf-8
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import image_provider
import matplotlib.pyplot as plt


tf.app.flags.DEFINE_string('buckets', './train/', '图片路径')
tf.app.flags.DEFINE_float('random_contrast_uper', 0.7, '随机对比度')
tf.app.flags.DEFINE_float('random_contrast_lower', 0.3, '随机对比度')
tf.app.flags.DEFINE_float('random_crop', 0.3, '随机裁剪')
tf.app.flags.DEFINE_float('random_scale', 0.3, '随机放大')
tf.app.flags.DEFINE_float('random_brightness', 73, '随机放大')
tf.app.flags.DEFINE_integer('batch_size', 1, '批大小')

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim


def main(args=None):
    reader = image_provider.Reader()
    image, label = reader.read()

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess=sess)


    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(
            image,
            num_classes=100,
            is_training=True)

    exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Conv2d_1a_3x3']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

    saver = tf.summary.FileWriter('./logs')
    saver.close()

if __name__ == '__main__':
    tf.app.run()
