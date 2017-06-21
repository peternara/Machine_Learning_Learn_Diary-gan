# coding=utf-8
import tensorflow as tf
import ops

flags = tf.flags.FLAGS


def discriminator(image):
    with tf.variable_scope('discriminator'):
        conv_1 = ops.conv_2d(image, 64, scope='conv_1')
        relu_1 = ops.leaky_relu(conv_1)

        conv_2 = ops.conv_2d(relu_1, 128, scope='conv_2')
        conv_2_norm = ops.batch_norm(conv_2, True)
        relu_2 = ops.leaky_relu(conv_2_norm)

        conv_3 = ops.conv_2d(relu_2, 256, scope='conv_3')
        conv_3_norm = ops.batch_norm(conv_3, True)
        relu_3 = ops.leaky_relu(conv_3_norm)

        conv_4 = ops.conv_2d(relu_3, 512, scope='conv_4')
        conv_4_norm = ops.batch_norm(conv_4, True)
        relu_4 = ops.leaky_relu(conv_4_norm)

        relu_4_flat = tf.reshape(relu_4, [flags.batch_size, -1])

        source_logits = ops.full_connect(relu_4_flat, 1, scope='source_logits')
        class_logits = ops.full_connect(relu_4_flat, 2, scope='class_logits')

        return source_logits, class_logits


def generator(z, label):
    with tf.variable_scope('generator'):
        label_one_hot = tf.one_hot(label, 2)
        z_label = tf.concat([z, label_one_hot], 1)

        image_width = flags.image_width
        image_height = flags.image_height

        z_label = ops.full_connect(z_label, 512 * image_width / 16 * image_height)  # 上采样 把label上采样到
        z_label = tf.reshape(z_label, [flags.bach_size, image_width / 16, image_height / 16, 512])
        z_label_norm = ops.batch_norm(z_label, True)
        relu_0 = tf.nn.relu(z_label_norm)

        decove_1 = ops.conv_2d_transpose(relu_0, [flags.batch_size, image_width / 8, image_height / 8, 256],
                                         scope='deconv_1')
        decove_1_norm = ops.batch_norm(decove_1, True)
        relu_1 = tf.nn.relu(decove_1_norm)

        decove_2 = ops.conv_2d_transpose(relu_1, [flags.batch_size, image_width / 4, image_height / 4, 128],
                                         scope='deconv_2')
        decove_2_norm = ops.batch_norm(decove_2, True)
        relu_2 = tf.nn.relu(decove_2_norm)

        decove_3 = ops.conv_2d_transpose(relu_2, [flags.batch_size, image_width / 2, image_height / 2, 64],
                                         scope='deconv_3')
        decove_3_norm = ops.batch_norm(decove_3, True)
        relu_3 = tf.nn.relu(decove_3_norm)

        deconv_4 = ops.conv_2d_transpose(relu_3, [flags.batch_size, image_width, image_height, 3], scope='deconv_4')
        tanh_4 = tf.nn.tanh(deconv_4)

        return tanh_4

