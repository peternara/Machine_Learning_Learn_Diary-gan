# coding=utf-8
import tensorflow as tf
import numpy as np
import ops
import layer

flags = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 20, """批大小""")
tf.flags.DEFINE_integer('image_height', 200, """图片高度""")
tf.flags.DEFINE_integer('image_width', 200, """图片宽度""")
tf.flags.DEFINE_integer('noise_size', 100, """噪声大小""")


def inference(image, label, z):
    generate_image = layer.generator(z, label)
    source_real_logits, source_real_classes = layer.discriminator(image)
    source_fake_logits, source_fake_classes = layer.discriminator(generate_image)

    return source_real_logits, source_real_classes, source_fake_logits, source_fake_classes


def main(args=None):
    z = tf.placeholder(tf.float32, [flags.batch_size, flags.noise_size], name="z")
    image = tf.placeholder(tf.float32, [flags.batch_size, flags.image_height, flags.image_width, 3])
    label = tf.placeholder(tf.uint8, [flags.batch_size, 2])


if __name__ == "__main__":
    tf.app.run()
