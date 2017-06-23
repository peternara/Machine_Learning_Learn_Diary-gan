# coding=utf-8
import tensorflow as tf
import ops

flags = tf.flags.FLAGS


def discriminator(image, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv_1 = ops.conv_2d(image, 64, scope='conv_1')
        relu_1 = ops.leaky_relu(conv_1)

        conv_2 = ops.conv_2d(relu_1, 128, scope='conv_2')
        conv_2_norm = ops.batch_norm(conv_2, True, scope="batch_norm_2")
        relu_2 = ops.leaky_relu(conv_2_norm)

        conv_3 = ops.conv_2d(relu_2, 256, scope='conv_3')
        conv_3_norm = ops.batch_norm(conv_3, True, scope="batch_norm_3")
        relu_3 = ops.leaky_relu(conv_3_norm)

        conv_4 = ops.conv_2d(relu_3, 512, scope='conv_4')
        conv_4_norm = ops.batch_norm(conv_4, True, scope="batch_norm_4")
        relu_4 = ops.leaky_relu(conv_4_norm)

        relu_4_flat = tf.reshape(relu_4, [flags.batch_size, -1])

        source_logits = ops.full_connect(relu_4_flat, 1, scope='source_logits')
        class_logits = ops.full_connect(relu_4_flat, 1, scope='class_logits')

        return source_logits, class_logits


def generator(z, label):
    with tf.variable_scope('generator'):
        label = tf.cast(label, tf.float32)
        z_label = tf.concat([z, label], 1)
        image_width = flags.image_width
        image_height = flags.image_height

        z_label = ops.full_connect(z_label, 512 * image_width / 16 * image_height / 16)  # 上采样 把label上采样
        z_label = tf.reshape(z_label, [flags.batch_size, image_width / 16, image_height / 16, 512])

        z_label_norm = ops.batch_norm(z_label, is_train=True)

        relu_0 = tf.nn.relu(z_label_norm)
        decove_1 = ops.conv_2d_transpose(relu_0, [flags.batch_size, image_width / 8, image_height / 8, 256],
                                         scope='deconv_1')
        decove_1_norm = ops.batch_norm(decove_1, True, scope="batch_norm_1")
        relu_1 = tf.nn.relu(decove_1_norm)

        decove_2 = ops.conv_2d_transpose(relu_1, [flags.batch_size, image_width / 4, image_height / 4, 128],
                                         scope='deconv_2')
        decove_2_norm = ops.batch_norm(decove_2, True, scope="batch_norm_2")
        relu_2 = tf.nn.relu(decove_2_norm)

        decove_3 = ops.conv_2d_transpose(relu_2, [flags.batch_size, image_width / 2, image_height / 2, 64],
                                         scope='deconv_3')
        decove_3_norm = ops.batch_norm(decove_3, True, scope="batch_norm_3")
        relu_3 = tf.nn.relu(decove_3_norm)

        deconv_4 = ops.conv_2d_transpose(relu_3, [flags.batch_size, image_width, image_height, 3], scope='deconv_4')
        tanh_4 = tf.nn.tanh(deconv_4)

        return tanh_4


def inference(image, label, z):
    generate_image = generator(z, label)
    real_logits, real_classes = discriminator(image)
    fake_logits, fake_classes = discriminator(generate_image, reuse=True)
    return real_logits, real_classes, fake_logits, fake_classes, generate_image


def loss(label, source_logits_real, class_logits_real, source_logits_fake, class_logits_fake):
    #  图片真假判断损失
    label = tf.cast(label, tf.float32)
    source_logits_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=source_logits_real,
            labels=tf.ones_like(source_logits_real)
        )
    )
    source_logits_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=source_logits_fake,
            labels=tf.zeros_like(source_logits_fake)
        )
    )

    #  生成模型损失
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=source_logits_fake,
            labels=tf.ones_like(source_logits_fake)
        )
    )

    #  类型判断损失
    class_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=class_logits_real,
            labels=label
        )
    )

    class_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=class_logits_fake,
            labels=label
        )
    )
    dc_loss = class_loss_real + class_loss_fake
    d_loss = source_logits_loss_real + source_logits_loss_fake + dc_loss
    g_loss = g_loss + dc_loss
    return d_loss, g_loss


def train(d_loss, g_loss):
    d_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"
    )
    g_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"
    )
    d_optimizer = tf.train.AdamOptimizer(flags.learn_rate, beta1=flags.beta1).minimize(d_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(flags.learn_rate, beta1=flags.beta1).minimize(g_loss, var_list=g_vars)
    return d_optimizer, g_optimizer
