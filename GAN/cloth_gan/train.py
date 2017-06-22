# coding=utf-8
import tensorflow as tf
import numpy as np
import layer
import read_records

flags = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 20, """批大小""")
tf.flags.DEFINE_integer('image_height', 200, """图片高度""")
tf.flags.DEFINE_integer('image_width', 200, """图片宽度""")
tf.flags.DEFINE_integer('noise_size', 100, """噪声大小""")
tf.flags.DEFINE_integer('train_steps', 1000, """训练次数""")
tf.flags.DEFINE_float('learn_rate', 3e-3, """训练次数""")


def main(args=None):
    z = tf.placeholder(tf.float32, [flags.batch_size, flags.noise_size], name="z")
    image = tf.placeholder(tf.float32, [flags.batch_size, flags.image_height, flags.image_width, 3])
    label = tf.placeholder(tf.uint8, [flags.batch_size, 2])

    source_logits_real, class_logits_real, source_logits_fake, class_logits_fake = layer.inference(image, label, z)

    d_loss, g_loss = layer.loss(label,
                                source_logits_real=source_logits_real, class_logits_real=class_logits_real,
                                source_logits_fake=source_logits_fake, class_logits_fake=class_logits_fake
                                )
    d_optimizer, g_optimazer = layer.train(d_loss, g_loss)

    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # tf.train.start_queue_runners(sess=sess)
    # saver = tf.train.Saver()
    # summary = tf.summary.FileWriter('logs')
    # reader = read_records.ReadRecords()
    # for step in xrange(flags.train_steps):
    #     random_z = np.random.uniform(-1, 1, size=[flags.batch_size, flags.noise_size]).astype(np.float32)
    #     image_read, label_read = reader.read()
    #     sess.run([d_optimizer, g_optimazer, g_optimazer], feed_dict={z: random_z, image: image_read, label: label_read})
    #     print "train steep: {}, d_loss: {}, g_loss: {}".format(step, d_loss, g_loss)


if __name__ == "__main__":
    tf.app.run()
