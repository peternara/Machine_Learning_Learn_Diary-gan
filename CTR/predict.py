# coding=utf-8
import os
import tensorflow as tf
import inference
import read

FLAGS = tf.flags.FLAGS

# buckets, checkpointDir, summaryDir都是规定的, 不建议更改

tf.flags.DEFINE_string('buckets', './data/', '数据源所在路径')
tf.flags.DEFINE_string('checkpointDir', './saves/', '模型保存路径')
tf.flags.DEFINE_string('summaryDir', './logs/', 'TensorBoard保存路径')
tf.flags.DEFINE_integer('hidden_1_size', 512, '隐藏层1神经元数')
tf.flags.DEFINE_integer('hidden_2_size', 256, '隐藏层2神经元数')
tf.flags.DEFINE_integer('output_size', 2, '输出数')

# 构造reader
reader = read.CTRReader(path=FLAGS.buckets, pattem='test.csv', is_training=False, num_classes=FLAGS.output_size)

# 获得数据和标签
datas, labels = reader.read()

# 构造神经网络
inference = inference.Inference(data_input=datas, h1_size=FLAGS.hidden_1_size, h2_size=FLAGS.hidden_2_size,
                                num_classes=FLAGS.output_size,
                                is_training=False)

logits = inference.get_softmax()

# 初始化
sess = tf.Session()
saver = tf.train.Saver()
tf.train.start_queue_runners(sess)
sess.run(tf.global_variables_initializer())
# 恢复保存的模型
saver.restore(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'CTR.model'))

# 计算AUC

auc = tf.contrib.metrics.streaming_auc(logits, labels)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(logits, 1)), tf.float32))
sess.run(tf.local_variables_initializer())

# 获取预测值
print "AUC: {}".format(sess.run(auc[1]))
print "accurcacy: {}".format(sess.run(accuracy))
