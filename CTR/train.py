# coding=utf-8
import os
import tensorflow as tf
import inference
import losses
import read

FLAGS = tf.flags.FLAGS

# buckets, checkpointDir, summaryDir都是规定的, 不建议更改

tf.flags.DEFINE_string('buckets', './data/', '数据源所在路径')
tf.flags.DEFINE_string('checkpointDir', './saves/', '模型保存路径')
tf.flags.DEFINE_string('summaryDir', './logs/', 'TensorBoard保存路径')
tf.flags.DEFINE_integer('batch_size', 50, '批大小')
tf.flags.DEFINE_integer('hidden_1_size', 512, '隐藏层1神经元数')
tf.flags.DEFINE_integer('hidden_2_size', 256, '隐藏层2神经元数')
tf.flags.DEFINE_integer('output_size', 2, '输出数')
tf.flags.DEFINE_integer('train_steps', 20000, '训练次数')
tf.flags.DEFINE_float('learning_rate', 1e-3, '学习速率')

# 构造reader
reader = read.CTRReader(batch_size=FLAGS.batch_size, path=FLAGS.buckets, num_classes=FLAGS.output_size,
                        pattem='train.*')

# 获得数据和标签
datas, labels = reader.read()

# 构造神经网络
inference = inference.Inference(data_input=datas, h1_size=FLAGS.hidden_1_size, h2_size=FLAGS.hidden_2_size,
                                num_classes=FLAGS.output_size,
                                is_training=True)
logits = inference.get_inference()
logits_softmax = inference.get_softmax()

# 构造损失
losses = losses.Losses(logits=logits, labels=labels).get_losses()

# 构造优化器
train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(losses)

# 初始化
sess = tf.Session()
summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
saver = tf.train.Saver(var_list=tf.trainable_variables())
tf.train.start_queue_runners(sess)
sess.run(tf.global_variables_initializer())

# 计算AUC

auc = tf.contrib.metrics.streaming_auc(logits_softmax, labels)
tf.summary.scalar('auc', auc[1])

sess.run(tf.local_variables_initializer())

# 迭代训练
for i in xrange(FLAGS.train_steps):


    sess.run(train_op)
    if i % 25 == 0 or i == FLAGS.train_steps - 1:
        summary_data = tf.summary.merge_all()
        summary.add_summary(sess.run(summary_data), i)
        print sess.run(auc[1])

saver.save(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'CTR.model'))
summary.close()
