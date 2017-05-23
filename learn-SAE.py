# coding: utf-8
import tensorflow as tf
import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder("float", shape=[None, 784])
# y_ = tf.placeholder("float", shape=[None, 10])

sess.run(tf.global_variables_initializer())

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])

h_pool1_flat = tf.reshape(h_pool1, [-1, 14, 14])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7, 7])

learning_rate = 0.001  # 学习速率
training_epochs = 5  # 训练批次
batch_size = 50  # 随机选择训练数据大小
display_step = 1  # 展示步骤
examples_to_show = 10  # 显示示例图片数量

# 网络参数
# 我这里采用了三层编码，实际针对mnist数据，隐层两层，分别为256，128效果最好
n_hidden_1 = 512  # 第一隐层神经元数量
n_hidden_2 = 256  # 第二
n_hidden_3 = 128  # 第三
n_input = 7 * 7 * 64  # 输入

# tf Graph输入
X = tf.placeholder("float", [None, n_input])

# 权重初始化
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

# 偏置值初始化
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}


# 开始编码
def encoder(x):
    # sigmoid激活函数，layer = x*weights['encoder_h1']+biases['encoder_b1']
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3


# 开始解码
def decoder(x):
    # sigmoid激活函数,layer = x*weights['decoder_h1']+biases['decoder_b1']
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3


# 构造模型
encoder_op = encoder(X)
encoder_result = encoder_op
decoder_op = decoder(encoder_op)

# 预测
y_pred = decoder_op
# 实际输入数据当作标签
y_true = X

# 定义代价函数和优化器，最小化平方误差,这里可以根据实际修改误差模型
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 初始化变量
sess.run(tf.global_variables_initializer())
# 总的batch
# total_batch = int(mnist.train.num_examples / batch_size)
# 开始训练
# for epoch in range(training_epochs):
#     for i in range(total_batch):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         cnn_out = sess.run(h_pool2_flat, feed_dict={x: batch_xs})
#         _, c = sess.run([optimizer, cost], feed_dict={X: x})
#         # 展示每次训练结果
#     if epoch % display_step == 0:
#         print("Epoch:", '%04d' % (epoch + 1),
#               "cost=", "{:.9f}".format(c))
# print("Optimization Finished!")
# # Applying encode and decode over test set
# # 显示编码结果和解码后结果
# encodes = sess.run(
#     encoder_result, feed_dict={X: mnist.test.images[:examples_to_show]})
# encode_decode = sess.run(
#     y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
# # 对比原始图片重建图片
# f, a = plt.subplots(2, 10, figsize=(10, 2))
# for i in range(examples_to_show):
#     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
#     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
# f.show()
# plt.draw()
# plt.waitforbuttonpress()
batch = mnist.train.next_batch(1)[0]
cnn_out = sess.run(h_pool1_flat, feed_dict={x:batch})
# ax.imshow(np.reshape(cnn_out[15], (14, 14)))
# bx.imshow(np.reshape(mnist.train.next_batch(1)[0], (28, 28)))
f, a = plt.subplots(2, 10, figsize=(64, 2))
f.set_figheight(50)
f.set_figwidth(50)
for key, i in enumerate(cnn_out[:10]):
    a[0][key].imshow(np.reshape(i, (14, 14)), cmap='gray')
    a[1][key].imshow(np.reshape(batch, (28, 28)), cmap='gray')
plt.draw()
plt.show()
