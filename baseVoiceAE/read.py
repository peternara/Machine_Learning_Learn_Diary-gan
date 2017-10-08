import tensorflow as tf


class Reader(object):
    def __init__(self, path, batch_size, window_size, kwidth):
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

        train_collect = []
        label_colloect = []
        for i in xrange(wav.shape[0] - kwidth - 1):
            train_collect.append(wav[i:i + kwidth])
            label_colloect.append(wav[i + kwidth])

        self.wav_raw, self.label = tf.train.shuffle_batch([train_collect, label_colloect], batch_size=batch_size,
                                                          num_threads=4,
                                                          capacity=1000 + 3 * batch_size,
                                                          min_after_dequeue=1000)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    reader = Reader('wavFile_train_frame_60.tfr', 1, 266, 32)
    tf.train.start_queue_runners()
    print reader.wav_raw
    layer = tf.expand_dims(tf.reshape(reader.wav_raw, [-1, 32]), 1)
    layer = tf.expand_dims(layer, 3)
    print layer
    data = layer.eval()
    print data[0]
    print data[1]
