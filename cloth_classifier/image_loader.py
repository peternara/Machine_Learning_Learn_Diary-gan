# coding=utf-8
import os
import tensorflow as tf

flag = tf.flags.FLAGS

tf.app.flags.DEFINE_string('train_path', './train',
                           """
                           训练集图片所在文件夹
                           """)
tf.app.flags.DEFINE_string('test_path', './test',
                           """
                           测试集图片所在文件夹
                           """)

tf.app.flags.DEFINE_integer('image_width', '50', """图片宽度""")
tf.app.flags.DEFINE_integer('image_height', '50', """图片高度""")
tf.app.flags.DEFINE_integer('batch_size', '30', """批大小""")
tf.app.flags.DEFINE_integer('test_size', '262', """批大小""")

tf.app.flags.DEFINE_integer('threads', '4', """线程数""")


def get_files(path):
    files = []
    for root, path, filename in os.walk(path):
        for name in filename:
            files.append(os.path.join(root, name))
    return files


def read_batch(path=flag.train_path, batch_size=flag.batch_size):
    """
    根据flag中的设置, 读取batch
    :returns:
    raws: 图片数据
    labels: 标签
    """
    files = get_files(path)
    file_queue = tf.train.string_input_producer(string_tensor=files, capacity=len(files))
    reader = tf.WholeFileReader()
    raws = []
    labels = []
    for i in xrange(batch_size):
        read = reader.read(file_queue)
        image = tf.image.decode_jpeg(read[1], channels=3)
        image = tf.image.resize_images(image, [flag.image_width, flag.image_height]) # 变形
        # image = tf.random_crop(image, [flag.image_width, flag.image_height, 3])  # 随机裁剪
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.per_image_standardization(image)
        raws.append(image)
        label = tf.string_split([read[0]], '/').values[2]
        label = tf.equal(tf.decode_raw(label, tf.uint8)[0], 84)
        label = tf.cast(label, tf.uint8)
        label = tf.one_hot(label, 2)
        labels.append(label)
    return raws, labels


def read_test(path=flag.test_path, batch_size=flag.test_size):
    """
    根据flag中的设置, 读取batch
    :returns:
    raws: 图片数据
    labels: 标签
    """
    files = get_files(path)
    file_queue = tf.train.string_input_producer(string_tensor=files, capacity=len(files))
    reader = tf.WholeFileReader()
    raws = []
    labels = []
    for i in xrange(batch_size):
        read = reader.read(file_queue)
        image = tf.image.decode_jpeg(read[1], channels=3)
        image = tf.image.resize_images(image, [flag.image_width, flag.image_height])  # 变形
        # image = tf.random_crop(image, [flag.image_width, flag.image_height, 3])  # 随机裁剪
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.per_image_standardization(image)
        raws.append(image)
        label = tf.string_split([read[0]], '/').values[2]
        label = tf.equal(tf.decode_raw(label, tf.uint8)[0], 84)
        label = tf.cast(label, tf.uint8)
        label = tf.one_hot(label, 2)
        labels.append(label)
    return raws, labels


def main(arg=None):
    sess = tf.Session()
    batch = read_batch()
    # batch = read_test()
    tf.train.start_queue_runners(sess=sess)
    print sess.run(batch)


if __name__ == '__main__':
    tf.app.run()
