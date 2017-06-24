# coding=utf-8
import tensorflow as tf

flags = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 20, """批大小""")
tf.flags.DEFINE_integer('image_height', 256, """图片高度""")
tf.flags.DEFINE_integer('image_width', 256, """图片宽度""")

class ReadRecords(object):
    def __init__(self,
                 train_path,
                 test_path,
                 batch_size=flags.batch_size,
                 width=flags.image_height,
                 height=flags.image_width,
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
            features = tf.parse_single_example(example, features={
                "label": tf.FixedLenFeature([], tf.int64),
                "image_raw": tf.FixedLenFeature([], tf.string)
            })
            image = tf.decode_raw(features['image_raw'], tf.float32)
            image = tf.reshape(image, [flags.image_height, flags.image_width, 3])
            label = features['label']
            images.append(image)
            labels.append([label])
        return images, labels


if __name__ == "__main__":
    reader = ReadRecords()
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)
    image, label = reader.read()
    print sess.run(label)
