# coding=utf-8
import tensorflow as tf
import numpy as np
import layer
import read_records
import os

flags = tf.flags.FLAGS
# tf.flags.DEFINE_integer('batch_size', 20, """批大小""")
# tf.flags.DEFINE_integer('image_height', 256, """图片高度""")
# tf.flags.DEFINE_integer('image_width', 256, """图片宽度""")
tf.flags.DEFINE_integer('noise_size', 500, """噪声大小""")
tf.flags.DEFINE_integer('train_steps', 5001, """训练次数""")
tf.flags.DEFINE_float('learn_rate', 1e-3, """训练速率""")
tf.flags.DEFINE_float('beta1', 0.5, """Adam 动量""")
tf.flags.DEFINE_string('buckets', None, """buckets""")
tf.flags.DEFINE_string('checkpointDir', None, """checkpointDir""")
tf.flags.DEFINE_string('summaryDir', None, """summaryDir""")


def main(args=None):
    z = tf.placeholder(tf.float32, [flags.batch_size, flags.noise_size], name="z")
    image = tf.placeholder(tf.float32, [flags.batch_size, flags.image_height, flags.image_width, 3])
    label = tf.placeholder(tf.uint8, [flags.batch_size, 1])

    source_logits_real, class_logits_real, source_logits_fake, class_logits_fake, generate_image = layer.inference(
        image, label, z)

    d_loss, g_loss = layer.loss(label,
                                source_logits_real=source_logits_real, class_logits_real=class_logits_real,
                                source_logits_fake=source_logits_fake, class_logits_fake=class_logits_fake
                                )
    d_optimizer, g_optimazer = layer.train(d_loss, g_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    reader = read_records.ReadRecords(train_path=flags.buckets + '*.train.tfrecords',
                                      test_path=flags.buckets + '*.test.tfrecords')
    tf.train.start_queue_runners(sess=sess)
    g_loss_save = tf.summary.scalar('g_loss', g_loss)
    d_loss_save = tf.summary.scalar('d_loss', d_loss)
    saver = tf.train.Saver(var_list=[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"),
                                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")],
                           keep_checkpoint_every_n_hours=0.5)
    # saver.restore(sess=sess, save_path=flags.checkpointDir)
    summary = tf.summary.FileWriter(flags.summaryDir)
    for step in xrange(flags.train_steps):
        random_z = np.random.uniform(-1, 1, size=[flags.batch_size, flags.noise_size]).astype(np.float32)
        image_read, label_read = sess.run(reader.read())
        sess.run([d_optimizer, g_optimazer, g_optimazer], feed_dict={z: random_z, image: image_read, label: label_read})

        if step % 50 == 0:
            image_save = tf.summary.image('generate_image', generate_image, max_outputs=50)
            merged = tf.summary.merge([image_save, g_loss_save, d_loss_save])
        else:
            merged = tf.summary.merge([g_loss_save, d_loss_save])
        merged = sess.run(merged, feed_dict={z: random_z, image: image_read, label: label_read})
        summary.add_summary(merged, step)
        print "train steep: {}".format(step)
        if step % 1000 == 0:
            saver.save(sess=sess, save_path=flags.checkpointDir, global_step=step)


if __name__ == "__main__":
    tf.app.run()
