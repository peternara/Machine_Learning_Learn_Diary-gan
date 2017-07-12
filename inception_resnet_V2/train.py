import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import inception_preprocessing

slim = tf.contrib.slim

image = inception_preprocessing('')

with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(
        [100, 299, 299, 3],
        num_classes=100,
        is_training=True)

exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Conv2d_1a_3x3']
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

sess = tf.InteractiveSession()
saver = tf.summary.FileWriter('./logs')
saver.close()
