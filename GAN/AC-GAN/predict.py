# coding=utf-8
import os

import numpy as np
import tensorflow as tf
import ac_gan

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('buckets', './image/', '图片文件夹')
tf.app.flags.DEFINE_string("checkpointDir", "./checkpoint_dir/", "模型保存路径")
tf.app.flags.DEFINE_bool('is_train', False, '是否在训练')
tf.app.flags.DEFINE_integer('num_classes', 133, '类型数')
tf.app.flags.DEFINE_integer('batch_size', 2, '批大小')


def perdict():
    writer = tf.gfile.GFile(FLAGS.checkpointDir + 'result.txt', 'wb')

    sess = tf.Session()
    reader = tf.WholeFileReader()

    files = tf.gfile.Glob(FLAGS.buckets + '*.jpg')
    file_queue = tf.train.string_input_producer(files, shuffle=False)

    file_name, file_content = reader.read(file_queue)
    image = tf.image.decode_jpeg(file_content, 3)
    image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
    image = tf.cast(image, tf.float32)
    images, file_names = tf.train.batch([image, file_name], batch_size=FLAGS.batch_size, num_threads=8)
    tf.train.start_queue_runners(sess=sess)

    classes = ac_gan.predict(images)

    saver = tf.train.Saver()
    ac_gan.load(sess, saver, checkpointDir=FLAGS.checkpointDir)

    for step in xrange(int(len(files) / FLAGS.batch_size)):
        print step
        image_per_batch, file_name_per_batch = sess.run([images, file_names])
        logits = sess.run(classes, feed_dict={images: image_per_batch})
        print logits
        for key, class_name in enumerate(logits):
            output = "{}\t{}\n".format(class_name, os.path.splitext(os.path.basename(file_name_per_batch[key]))[0])
            writer.write(output)
    writer.close()


def main(argv=None):
    for i in FLAGS.__flags:
        print "{}: {}".format(i, FLAGS.__flags[i])
    perdict()


if __name__ == '__main__':
    tf.app.run()
