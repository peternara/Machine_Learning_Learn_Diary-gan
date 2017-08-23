import tensorflow as tf
import numpy as np
import os
import read
import model_vae
import time

#图级随机seed
np.random.seed(0)
tf.set_random_seed(0)


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('buckets','D:/Documents/','shujuyuan')
tf.flags.DEFINE_string('checkpointDir', 'D:/Documents/temp/', '模型保存路径')
tf.flags.DEFINE_string('summaryDir', 'D:/Documents/temp//logs/', 'TensorBoard保存路径')
tf.flags.DEFINE_integer('batch_size', 100, '批大小')
tf.flags.DEFINE_integer('display_step', 5, '显示步数')
tf.flags.DEFINE_float('learning_rate', 1e-3, '学习速率')
tf.flags.DEFINE_float('training_epochs', 1000, '训练次数')

train_file_path = os.path.join(FLAGS.buckets, "train.tfrecords")
test_file_path = os.path.join(FLAGS.buckets, "test.tfrecords")

#下载数据
reader = read.MnistReader(train_file_path)
mnist = reader.read() 
n_samples =  mnist.shape[0]       
  

vae = model_vae.VariationalAutoencoder(learning_rate=FLAGS.learning_rate, 
                             batch_size=FLAGS.batch_size)
# 创建自编码器网络
print("创建变分自编码器网络")
vae._create_network()

# 损失函数和优化器
print("构造损失函数和优化器")
vae._create_loss_optimizer()


sess = tf.InteractiveSession()
summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
saver = tf.train.Saver(var_list=tf.trainable_variables())
tf.global_variables_initializer().run() 
                           

for epoch in range(FLAGS.training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples/ FLAGS.batch_size)
    for i in range(total_batch):
        batch_xs= mnist[(i*FLAGS.batch_size):((i+1)*FLAGS.batch_size),:]       
        _,cost = sess.run(fetches = [vae.optimizer,vae.cost],feed_dict={vae.x: batch_xs})
        avg_cost += cost / n_samples * FLAGS.batch_size
        if i % 5000 == 0:
            summary_data = tf.summary.merge_all()
            summary.add_summary(sess.run(summary_data,feed_dict={vae.x: batch_xs}), epoch*550 + i/5000 )            
    if epoch % FLAGS.display_step == 0:
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print("Epoch:", '%04d' % (epoch+1), 
              "cost=", "{:.9f}".format(avg_cost))

saver.save(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'vae.model'))          

summary.close()
sess.close()
print("训练成功")

