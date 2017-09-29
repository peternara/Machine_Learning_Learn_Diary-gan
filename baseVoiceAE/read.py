import tensorflow as tf
import numpy as np


class Reader(object):
    def __init__(self, sess, path, batch_size, window_size, kwidth):
        self.sess = sess
        self.kwidth = kwidth
        self.reader = tf.TFRecordReader()
        self.file_queue = tf.train.string_input_producer([path])
        _, example = self.reader.read(self.file_queue)
        features = tf.parse_single_example(
            example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string)
            })

        wav = tf.decode_raw(features['wav_raw'], tf.int16)
        wav.set_shape([window_size])
        wav = tf.cast(wav, tf.float32)
        wav = (2. / 65535.) * tf.cast((wav - 32767), tf.float32) + 1.
        zero_padding = tf.constant(np.zeros([kwidth], dtype=np.float32))
        wav = tf.concat([zero_padding, wav], 0)
        self.wav_raw = tf.train.shuffle_batch([wav], batch_size=batch_size, num_threads=4,
                                              capacity=1000 + 3 * batch_size,
                                              min_after_dequeue=1000)

    def read(self):
        wavs_raw = self.sess.run(self.wav_raw)
        wavs = []
        labels = []
        for wav in wavs_raw:
            for index in xrange(0, len(wav)):
                part_wavs = wav[index:index + self.kwidth]
                if not len(part_wavs) == 18:
                    continue
                try:
                    part_label = wav[index + self.kwidth + 1]
                except IndexError:
                    continue
                wavs.append(part_wavs)
                labels.append([part_label])
        return wavs, labels


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    reader = Reader(sess, 'wavFile_train_frame_60.tfr', 2, 266, 18)
    tf.train.start_queue_runners()
    print reader.read()
