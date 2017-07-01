import os

import tensorflow as tf
import scipy.misc
import warnings

files = tf.gfile.Glob('./saves/*.jpg')

reader = tf.WholeFileReader()

sess = tf.Session()

for file in files:
    try:
        image = scipy.misc.imread(file)
    except Warning, e:
        print "warning: {}, {} has broken".format(e.message, file)
        continue
    except Exception, e:
        print "error: {}, {} has broken".format(e.message, file)
        continue
    else:
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)

        label = os.path.split(file)[-1].split('-')[0]
        print int(label)

        writer = tf.python_io.TFRecordWriter('./Records')


