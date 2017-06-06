# coding=utf-8
import tensorflow as tf
import image_loader
import inference
import loss as ls

flag = tf.flags.FLAGS

tf.flags.DEFINE_string('save_dir', './saves/', """保存路径""")
tf.flags.DEFINE_string('log_dir', './logs/', """logs路径""")


def main(arg=None):
    images, labels = image_loader.read_test()
    logits = inference.inference(images)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()

    sess.run(init)
    saver.restore(sess, flag.save_dir)
    tf.train.start_queue_runners(sess=sess)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print sess.run(accuracy)


if __name__ == '__main__':
    tf.app.run()
