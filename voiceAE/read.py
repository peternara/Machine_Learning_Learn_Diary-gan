import tensorflow as tf
import os


class Reader(object):
    def __init__(self, sess, path, batch_size, canvas_size):
        self.sess = sess
        with tf.name_scope('read_and_parse'):
            reader = tf.TFRecordReader()
            files = tf.gfile.Glob(os.path.join(path, "*.tfrecords"))
            file_queue = tf.train.string_input_producer(files)
            _, example = reader.read(file_queue)

            features = tf.parse_single_example(
                example,
                features={
                    'wav_raw': tf.FixedLenFeature([], tf.string),
                    'noisy_raw': tf.FixedLenFeature([], tf.string),
                })

        with tf.name_scope('decode_and_norm'):
            clear = tf.decode_raw(features['wav_raw'], tf.int32)
            clear.set_shape(canvas_size)
            clear = (2. / 65535.) * tf.cast((clear - 32767), tf.float32) + 1.

            noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
            noisy.set_shape(canvas_size)
            noisy = (2. / 65535.) * tf.cast((noisy - 32767), tf.float32) + 1.

        self.wav_raw, self.noisy_raw = tf.train.shuffle_batch([clear, noisy], batch_size=batch_size, num_threads=4,
                                                              capacity=1000 + 3 * batch_size,
                                                              min_after_dequeue=1000, name='wav_and_noisy')

    def read(self):
        return self.wav_raw, self.noisy_raw


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    reader = Reader(sess, 'data', 64, 2 ** 14)
    tf.train.start_queue_runners()
    wav, noise = reader.read()
    # import loss
    #
    # loss = loss.Losses(wav, noise).get_loss()
    #
    # print loss.eval()
    print wav.eval()

