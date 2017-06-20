import tensorflow as tf

class ReadRecords(object):
    def __init__(self,
                 train_path='./tfRecords/0.train.tfrecords',
                 test_path='./tfRecords/*.test.tfrecords',
                 batch_size=20,
                 width=200,
                 height=200,
                 isTrain=True,
                 shuffer=True):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.shuffer = shuffer
        self.isTrain = isTrain
        self.height = height
        self.width = width
        file_name = tf.gfile.Glob(train_path)
        self.file_queue = tf.train.string_input_producer(file_name)

    def read(self):
        reader = tf.TFRecordReader()
        images = []
        labels = []
        for i in xrange(self.batch_size):
            _, example = reader.read(queue=self.file_queue)
            features = tf.parse_example(example, features={
                "label": tf.FixedLenFeature([], tf.int64),
                "image_raw": tf.FixedLenFeature([], tf.string)
            })
            image = tf.decode_raw(features['image_raw'], tf.float32)
            image = tf.reshape(image, [200, 200, 3])
            label = features['label']
            images.append(image)
            labels.append(label)
        return images, labels