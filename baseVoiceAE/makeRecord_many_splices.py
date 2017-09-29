import tensorflow as tf
import scipy.io.wavfile as wavfile

frame_count = 60
kwidth = 18

writer_train = tf.python_io.TFRecordWriter('wavFile_train_frame_{}.tfr'.format(frame_count))
writer_test = tf.python_io.TFRecordWriter('wavFile_test_frame_{}.tfr'.format(frame_count))

for file_path in tf.gfile.Glob('data/*.WAV'):
    fm, wav = wavfile.read(file_path)

    window = fm // frame_count

    print "Parsing file: {}, window size: {}, kwidth: {}".format(file_path, window, kwidth)

    for i in xrange(0, fm, window):
        signal = wav[i:window + i]
        for index in signal:
            part_signal = signal[index:index + kwidth]
            if not len(part_signal) == 18:
                continue
            try:
                part_label = signal[index + kwidth + 1]
            except IndexError:
                continue
            if not i == window * frame_count:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'wav_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[part_signal.tostring()])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[part_label.tostring()]))
                }))
                writer_train.write(example.SerializeToString())
            else:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'wav_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[part_signal.tostring()])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[part_label.tostring()]))
                }))
                writer_test.write(example.SerializeToString())
