import os

import tensorflow as tf
import scipy.misc
import numpy as np

flags = tf.flags.FLAGS
# tf.app.flags.DEFINE_string('buckets', './saves/', 'buckets')
tf.app.flags.DEFINE_string('pattem', '*.jpg', 'pattem')
tf.app.flags.DEFINE_string('checkpointDir', './Records/', 'checkpointDir')

image_path = './saves/'


def main(arg=None):
    if not tf.gfile.Exists(os.path.join(flags.checkpointDir)):
        tf.gfile.MkDir(os.path.join(flags.checkpointDir))

    files = tf.gfile.Glob(os.path.join(image_path, flags.pattem))

    sess = tf.Session()
    for key, file in enumerate(files):
        print key, file
        if tf.gfile.Exists(os.path.join(flags.checkpointDir, os.path.basename(file)) + '.tfrecords'):
            print "{} already Exist, skip!".format(file)
            continue
        try:
            image = scipy.misc.imread(file)
            image_shape = np.shape(image)
            if not len(image_shape) == 3:
                print "error: {} is bad image".format(file)
                continue
            else:
                image = image[:, :, 0:3]
        except Warning, e:
            print "warning: {}, {} has broken will be Delete".format(e.message, file)
            # os.remove(file)
            continue
        except Exception, e:
            print "error: {}, {} has broken will be Delete".format(e.message, file)
            # os.remove(file)
            continue
        else:
            image = tf.image.per_image_standardization(image)
            image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)

            label = os.path.split(file)[-1].split('-')[0]

            image = sess.run(image).tostring()
            label = int(label)

            writer = tf.python_io.TFRecordWriter(
                os.path.join(flags.checkpointDir, '{}.tfrecords'.format(os.path.basename(file))))

            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

            writer.write(example.SerializeToString())
            writer.close()
            # os.remove(file)


if __name__ == '__main__':
    tf.app.run()
