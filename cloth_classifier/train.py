# coding=utf-8
import tensorflow as tf
import image_loader
import inference
import loss as ls


flag = tf.flags.FLAGS

tf.flags.DEFINE_string('save_dir', './saves/', """保存路径""")
tf.flags.DEFINE_string('log_dir', './logs/', """logs路径""")


def main(arg=None):
    images, labels = image_loader.read_batch()
    logits = inference.inference(images)
    loss = ls.loss(logits, labels)

    saver = tf.train.Saver()

    summary_opt = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()

    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(flag.log_dir, graph=sess.graph)

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in xrange(5001):
        if i % 100 == 0:
            print 'step {0}, loss: {1}'.format(i, sess.run(ls.get_loss()))
        sess.run(train_step)
        if i % 50 == 0:
            summary_str = sess.run(summary_opt)
            summary_writer.add_summary(summary_str, i)

    saver.save(sess=sess, save_path=flag.save_dir)
    summary_writer.close()



if __name__ == '__main__':
    tf.app.run()
