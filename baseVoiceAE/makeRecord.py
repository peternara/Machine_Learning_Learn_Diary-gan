import tensorflow as tf
import scipy.io.wavfile as wavfile

frame_count = 60

writer_train = tf.python_io.TFRecordWriter('wavFile_train_frame_{}.tfr'.format(frame_count))
writer_test = tf.python_io.TFRecordWriter('wavFile_test_frame_{}.tfr'.format(frame_count))

for file_path in tf.gfile.Glob('data/*.WAV'):
    fm, wav = wavfile.read(file_path)

    window = fm // frame_count

    print "Parsing file: {}, window size: {}".format(file_path, window)

    for i in xrange(0, fm, window):
        signal = wav[i:window + i]
        if not i == window * frame_count:
            example = tf.train.Example(features=tf.train.Features(feature={
                'wav_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[signal.tostring()]))
            }))
            writer_train.write(example.SerializeToString())
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'wav_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[signal.tostring()]))
            }))
            writer_test.write(example.SerializeToString())
