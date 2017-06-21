import tensorflow as tf

class TFRecord_Generator(object):
    def __init__(self,
                 train_path='train/*.jpg',
                 test_path='test/*.jpg',
                 isTrain=True,
                 crop_resize=True,
                 standarlizing=True,
                 width=200,
                 height=200,
                 ):
        if isTrain:
            file_names = tf.gfile.Glob(train_path)
        else:
            file_names = tf.gfile.Glob(test_path)
        file_names_queue = tf.train.string_input_producer(file_names)
        reader = tf.WholeFileReader()
        self.file_queue = file_names_queue
        self.reader = reader
        self.train_path = train_path
        self.test_path = test_path
        self.crop_resize = crop_resize
        self.standarlizing = standarlizing
        self.width = width
        self.height = height
        self.file_count = len(file_names)
        self.isTrain = isTrain

    def __read(self):
        label, image = self.reader.read(self.file_queue)
        image = tf.image.decode_jpeg(image, 3)
        if self.standarlizing:
            image = tf.image.per_image_standardization(image)
        if self.crop_resize:
            image = tf.image.resize_image_with_crop_or_pad(image, self.width, self.height)
        else:
            image = tf.image.resize_images(image, [self.width, self.height])
        label = tf.string_split([label], '/').values[1]
        label = tf.equal(tf.decode_raw(label, tf.uint8)[0], 84)
        label = tf.cast(label, tf.uint8)
        return image, label

    def __write_tfrecode(self, writer, image, label):
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))
        writer.write(example.SerializeToString())

    def generate(self, sess):
        tf.train.start_queue_runners(sess=sess)
        for i in xrange(self.file_count):
            image, label = sess.run(self.__read())
            if self.isTrain:
                train_bath = "train"
            else:
                train_bath = "test"
            writer = tf.python_io.TFRecordWriter('./tfRecords/{}.{}.tfrecords'.format(i, train_bath))
            self.__write_tfrecode(writer, image.tostring(), label)
            print "reading {}".format(i)

    def generate_test(self, sess):
        tf.train.start_queue_runners(sess=sess)


if __name__ == "__main__":
    generator = TFRecord_Generator()
    with tf.Session() as sess:
        generator.generate(sess)
        # generator = TFRecord_Generator(isTrain=False)
        # with tf.Session() as sess:
        #     generator.generate(sess)
