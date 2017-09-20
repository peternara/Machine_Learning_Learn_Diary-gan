import tensorflow as tf


class Reader(object):
    def __init__(self, path, batch_size, window_size):
        self.reader = tf.TFRecordReader()
        self.file_queue = tf.train.string_input_producer([path])
        _, example = self.reader.read(self.file_queue)
        features = tf.parse_single_example(
            example,
            # Defaults are not specified since both keys are required.
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string)
            })
        wav = tf.decode_raw(features['wav_raw'], tf.int16)
        wav.set_shape([window_size])
        self.wav_raw = tf.train.shuffle_batch([wav], batch_size=batch_size, num_threads=4,
                                              capacity=1000 + 3 * batch_size,
                                              min_after_dequeue=1000)

    def read(self):
        return self.wav_raw


if __name__ == '__main__':
    reader = Reader('wavFile.tfr', 2)
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners()
    print sess.run(reader.read())
