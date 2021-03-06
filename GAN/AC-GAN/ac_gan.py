# coding=utf-8
import os

import tensorflow as tf

import ops

tf.app.flags.DEFINE_integer('input_height', 256, '输入图片高度')
tf.app.flags.DEFINE_integer('input_width', 256, '输入图片宽度')
tf.app.flags.DEFINE_integer('input_channels', 3, '图片通道')
tf.app.flags.DEFINE_integer('output_height', 128, '输出图片高度')
tf.app.flags.DEFINE_integer('output_width', 128, '输出图片宽度')
tf.app.flags.DEFINE_integer('z_dim', 100, '噪音数目')
FLAGS = tf.app.flags.FLAGS


def inference(images, labels, z):
    generated_images = generator(z, labels)
    source_logits_real, class_logits_real = discriminator(images)
    source_logits_fake, class_logits_fake = discriminator(generated_images, reuse=True)

    return [
        source_logits_real, class_logits_real, source_logits_fake,
        class_logits_fake, generated_images
    ]


def predict(images):
    source_logits_real, class_logits_real = discriminator_predict(images)
    class_logits = tf.arg_max(class_logits_real, 1)
    return class_logits


def loss(labels,
         source_logits_real,
         class_logits_real,
         source_logits_fake,
         class_logits_fake,
         generated_images):
    #   判断图片真假损失
    # source_loss_real = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=source_logits_real,
    #         labels=tf.ones_like(source_logits_real)
    #     ))
    #
    # source_loss_fake = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=source_logits_fake,
    #         labels=tf.zeros_like(source_logits_fake)))
    #  生成图片损失
    # g_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=source_logits_fake,
    #         labels=tf.ones_like(source_logits_fake)))
    #   判断图片类别损失
    class_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=class_logits_real,
            labels=labels))
    class_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=class_logits_fake,
            labels=labels))

    l2_loss = tf.add_n(tf.get_collection('l2_loss'))
    dc_loss = class_loss_real + class_loss_fake + l2_loss

    d_loss = -tf.reduce_mean(source_logits_real) - tf.reduce_mean(source_logits_real)
    g_loss = -tf.reduce_mean(source_logits_fake)
    return d_loss, g_loss, dc_loss


def train(d_loss, g_loss):
    # variables for discriminator
    d_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # variables for generator
    g_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    # train discriminator
    d_optimzer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    train_d_op = d_optimzer.minimize(-d_loss, var_list=d_vars)

    # train generator
    g_optimzer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    train_g_op = g_optimzer.minimize(g_loss, var_list=g_vars)

    return train_d_op, train_g_op


def discriminator(images, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        # conv1
        conv1 = ops.conv_2d(images, 64, scope="conv1")

        # leakly ReLu
        h1 = ops.leaky_relu(conv1)

        # conv2
        conv2 = ops.conv_2d(h1, 128, scope="conv2")

        # batch norm
        norm2 = ops.batch_norm(conv2, scope="batch_norm2", is_training=FLAGS.is_train)

        # leaky ReLU
        h2 = ops.leaky_relu(norm2)

        # conv3
        conv3 = ops.conv_2d(h2, 256, scope="conv3")
        # batch norm
        norm3 = ops.batch_norm(conv3, scope="batch_norm3", is_training=FLAGS.is_train)

        # leaky ReLU
        h3 = ops.leaky_relu(norm3)
        # conv4
        conv4 = ops.conv_2d(h3, 512, scope="conv4")

        # batch norm
        norm4 = ops.batch_norm(conv4, scope="batch_norm4", is_training=FLAGS.is_train)

        # leaky ReLU
        h4 = ops.leaky_relu(norm4)

        conv5 = ops.conv_2d(h4, 1024, scope="conv5")

        conv5 = tf.nn.dropout(conv5, 0.5, name='conv_5_drop_out')

        norm5 = ops.batch_norm(conv5, scope="batch_norm5", is_training=FLAGS.is_train)

        h5 = ops.leaky_relu(norm5)
        # reshape
        h5_reshape = tf.reshape(h5, [FLAGS.batch_size, -1])

        # source logits
        source_logits = ops.fc(h5_reshape, 1, scope="source_logits")

        # class logits
        class_logits = ops.fc(
            h5_reshape, FLAGS.num_classes, scope="class_logits", decay=4e-3)

        return source_logits, class_logits


def discriminator_predict(images, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        # conv1
        conv1 = ops.conv_2d(images, 64, scope="conv1")

        # leakly ReLu
        h1 = ops.leaky_relu(conv1)

        # conv2
        conv2 = ops.conv_2d(h1, 128, scope="conv2")

        # batch norm
        norm2 = ops.batch_norm(conv2, scope="batch_norm2", is_training=FLAGS.is_train)

        # leaky ReLU
        h2 = ops.leaky_relu(norm2)

        # conv3
        conv3 = ops.conv_2d(h2, 256, scope="conv3")
        # batch norm
        norm3 = ops.batch_norm(conv3, scope="batch_norm3", is_training=FLAGS.is_train)

        # leaky ReLU
        h3 = ops.leaky_relu(norm3)
        # conv4
        conv4 = ops.conv_2d(h3, 512, scope="conv4")

        # batch norm
        norm4 = ops.batch_norm(conv4, scope="batch_norm4", is_training=FLAGS.is_train)

        # leaky ReLU
        h4 = ops.leaky_relu(norm4)

        conv5 = ops.conv_2d(h4, 1024, scope="conv5")

        norm5 = ops.batch_norm(conv5, scope="batch_norm5", is_training=FLAGS.is_train)

        h5 = ops.leaky_relu(norm5)
        # reshape
        h5_reshape = tf.reshape(h5, [FLAGS.batch_size, -1])

        # source logits
        source_logits = ops.fc(h5_reshape, 1, scope="source_logits")

        # class logits
        class_logits = ops.fc(
            h5_reshape, FLAGS.num_classes, scope="class_logits", decay=4e-3)

        return source_logits, class_logits


def generator(z, labels):
    with tf.variable_scope("generator") as scope:
        # concat z and labels
        z_labels = tf.concat([z, labels], 1)
        # project z and reshape
        oh, ow = FLAGS.output_height, FLAGS.output_width

        z_labels_ = ops.fc(z_labels, 1024 * oh / 32 * ow / 32, scope="project")
        z_labels_ = tf.reshape(z_labels_, [-1, oh / 32, ow / 32, 1024])

        # batch norm
        norm0 = ops.batch_norm(
            z_labels_, scope="batch_norm0", is_training=True)

        # ReLU
        h0 = tf.nn.relu(norm0)

        # conv1
        conv1 = ops.conv2d_transpose(
            h0, [FLAGS.batch_size, oh / 16, ow / 16, 512],
            scope="conv_tranpose1")

        # batch norm
        norm1 = ops.batch_norm(conv1, scope="batch_norm1", is_training=FLAGS.is_train)

        # ReLU
        h1 = tf.nn.relu(norm1)

        # conv2
        conv2 = ops.conv2d_transpose(
            h1, [FLAGS.batch_size, oh / 8, ow / 8, 256],
            scope="conv_tranpose2")

        # batch norm
        norm2 = ops.batch_norm(conv2, scope="batch_norm2", is_training=FLAGS.is_train)

        # ReLU
        h2 = tf.nn.relu(norm2)

        # conv3
        conv3 = ops.conv2d_transpose(
            h2, [FLAGS.batch_size, oh / 4, ow / 4, 128], scope="conv_tranpose3")

        # batch norm
        norm3 = ops.batch_norm(conv3, scope="batch_norm3", is_training=FLAGS.is_train)

        # ReLU
        h3 = tf.nn.relu(norm3)

        # conv4
        conv4 = ops.conv2d_transpose(
            h3, [FLAGS.batch_size, oh / 2, ow / 2, 64],
            scope="conv_tranpose4")
        # batch norm
        norm4 = ops.batch_norm(conv4, scope="batch_norm4", is_training=FLAGS.is_train)

        # ReLU
        h4 = tf.nn.relu(norm4)

        conv5 = ops.conv2d_transpose(
            h4, [FLAGS.batch_size, oh, ow, FLAGS.input_channels],
            scope="conv_tranpose5")
        # tanh
        h5 = tf.nn.tanh(conv5)

        h5 = tf.map_fn(lambda i: tf.image.resize_images(i, [FLAGS.input_height, FLAGS.input_width]), h5)

    return h5


def load(sess, saver, checkpointDir):
    if tf.train.get_checkpoint_state(checkpointDir):
        latest_checkpoint = os.path.basename(tf.train.latest_checkpoint(FLAGS.checkpointDir))
        saver.restore(sess, os.path.join(checkpointDir, latest_checkpoint))
        print "checkpoint get"
    else:
        print "checkpoint not found"


class Reader:
    def __init__(self, path, pattem, batch_size, num_classes):
        files = tf.gfile.Glob(os.path.join(path, pattem))
        self.batch_size = batch_size
        self.fileQueue = tf.train.string_input_producer(files, shuffle=True)
        self.reader = tf.TFRecordReader()
        self.num_classes = num_classes

    def read(self):
        labels = []
        images = []
        with tf.name_scope('reader_scope'):
            for key in xrange(self.batch_size):
                _, content = self.reader.read(self.fileQueue)
                example = tf.parse_single_example(content, features={
                    'label': tf.FixedLenFeature([], tf.int64),
                    'image': tf.FixedLenFeature([], tf.string)
                })
                image = tf.decode_raw(example['image'], tf.float32)
                image = tf.reshape(image, [FLAGS.input_height, FLAGS.input_width, FLAGS.input_channels])
                label = example['label']
                label = tf.one_hot(label, self.num_classes)
                labels.append(label)
                images.append(image)

        return labels, images


if __name__ == '__main__':
    reader = Reader('Records', '*.tfrecords', 16, 113)
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess=sess)
    labels, images = reader.read()
    print images[0].get_shape()
    # print sess.run(labels)
