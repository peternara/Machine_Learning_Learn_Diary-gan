# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import read
import model_vae
import time

# 图级随机seed
np.random.seed(0)
tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('buckets', './data/', 'shujuyuan')
tf.flags.DEFINE_string('checkpointDir', './saves/', '模型保存路径')
tf.flags.DEFINE_string('summaryDir', './logs/', 'TensorBoard保存路径')
tf.flags.DEFINE_integer('batch_size', 1, '批大小')
tf.flags.DEFINE_integer('display_step', 5, '显示步数')
tf.flags.DEFINE_float('learning_rate', 1e-3, '学习速率')
tf.flags.DEFINE_float('training_epochs', 500, '训练次数')

train_file_path = os.path.join(FLAGS.buckets, "train.tfrecords")
test_file_path = os.path.join(FLAGS.buckets, "test.tfrecords")

# #reader = read.MnistReader(train_file_path)
# mnist = reader.read() 
# n_samples =  mnist.shape[0]       
# print(mnist.shape)    

vae = model_vae.VariationalAutoencoder(learning_rate=FLAGS.learning_rate,
                                       batch_size=FLAGS.batch_size)
# 创建自编码器网络
print("创建变分自编码器网络")
vae._create_network()

# 损失函数和优化器
# print("损失函数和优化器")
# vae._create_loss_optimizer()


sess = tf.InteractiveSession()
summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
saver = tf.train.Saver(var_list=tf.trainable_variables())
saver.restore(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'vae.model'))

nx = ny = 20
# -2到2 nx个数的等差数列
# x_values = np.hstack( (np.linspace(-2, -1, 2) ,np.linspace(-1, 1, nx-4),np.linspace(-2, 2, 2)) )
# y_values = np.hstack( (np.linspace(-2, -1, 2) ,np.linspace(-1, 1, ny-4),np.linspace(-2, 2, 2)) )
x_values = np.linspace(-2, 2, nx)
y_values = np.linspace(-2, 2, ny)

canvas = np.empty((28 * ny, 28 * nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi]] * FLAGS.batch_size)
        x_mean = sess.run(vae.x_reconstr_mean, feed_dict={vae.z: z_mu})
        canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean.reshape(28, 28)

batch_xs = canvas * 255
batch_xs = np.asarray(batch_xs, dtype='uint8')
image_summary = tf.summary.image('gen_image', tf.expand_dims(tf.expand_dims(batch_xs, 0), 3))

summary_data = tf.summary.merge_all()
summary.add_summary(sess.run(image_summary), 0)
# saver.save(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'vae.model'))
summary.close()
print("图像生成成功")
