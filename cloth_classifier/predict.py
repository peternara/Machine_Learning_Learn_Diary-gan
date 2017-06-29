# coding=utf-8
import shutil

import tensorflow as tf
import image_loader
import inference_predict as inference

flag = tf.flags.FLAGS

tf.flags.DEFINE_string('save_dir', './saves/', """保存路径""")
tf.flags.DEFINE_string('log_dir', './logs/', """logs路径""")


def main(arg=None):
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    files = image_loader.get_files('/Users/wanqianjun/Desktop/python/淘宝客爬虫/img_save')
    queue, reader = image_loader.read_queue(files)
    tf.train.start_queue_runners(sess=sess)
    filepath, image = reader.read(queue)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [flag.image_width, flag.image_height])
    image = tf.image.per_image_standardization(image)
    logits = tf.nn.softmax(inference.inference([image]))
    logits = tf.arg_max(logits, 1)
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, flag.save_dir)
    for key in xrange(len(files)):
        try:
            persentage, path = sess.run([logits, filepath])
            if persentage[0]:
                shutil.copyfile(path, './img_save/T{}_new.jpg'.format(key))
            print key
        except Exception,e:
            print e.message
            continue


if __name__ == '__main__':
    tf.app.run()
