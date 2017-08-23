import tensorflow as tf
import numpy as np
import os
import time

#图级随机seed
np.random.seed(0)
tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('buckets','D:/Documents/','模型保存路径')
tf.flags.DEFINE_string('checkpointDir', 'D:/Documents/temp/', '模型保存路径')
tf.flags.DEFINE_string('summaryDir', 'D:/Documents/temp/', 'TensorBoard保存路径')
tf.flags.DEFINE_integer('batch_size', 100, '批大小')
tf.flags.DEFINE_float('learning_rate', 1e-3, '学习速率')
tf.flags.DEFINE_float('training_epochs', 400*550, '训练次数')

train_file_path = os.path.join(FLAGS.buckets, "train.tfrecords")
test_file_path = os.path.join(FLAGS.buckets, "test.tfrecords")




###########################################################神经网络结构         
def xavier_init(shape_in,shape_out, constant=1):
    low = -constant*np.sqrt(6.0/(shape_in + shape_out)) 
    high = constant*np.sqrt(6.0/(shape_in + shape_out))

    return tf.Variable(tf.random_uniform((shape_in, shape_out),minval = low, maxval=high,dtype=tf.float32))

def bias(shape):
    return tf.constant(0.1, shape=shape)

class VariationalAutoencoder(object):

    def __init__(self,input_x, transfer_fct=tf.nn.relu , #softplus relu  sigmoid, softplus  ,  lrelu 
                    n_hidden_recog_1 = 500, n_hidden_recog_2 = 500, 
                    n_hidden_gener_1 = 500,  n_hidden_gener_2 = 500, 
                    n_input = 784, n_z = 2,
                    learning_rate=0.001, batch_size=100):
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_hidden_recog_1 = n_hidden_recog_1
        self.n_hidden_recog_2 = n_hidden_recog_2
        self.n_hidden_gener_1 = n_hidden_gener_1
        self.n_hidden_gener_2 = n_hidden_gener_2
        self.n_input = n_input
        self.n_z = n_z
        
        self.x = input_x
        # # 创建自编码器网络
        # self._create_network()
        
        # # 损失函数和优化器
        # self._create_loss_optimizer()
        
    
    def _create_network(self):
        self.z_mean, self.z_log_sigma_sq = self._recognition_network()

        eps = tf.random_normal((self.batch_size, self.n_z), 0, 1,  dtype=tf.float32)
        with tf.variable_scope('z'): 
            # z = mu + sigma*epsilon
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
            self._summary_helper(self.z)

        self.x_reconstr_mean = self._generator_network()
            

 
    #编码网络encode network 
    def _recognition_network(self):
        with tf.variable_scope('recognition_network_layer_1'): 
            rec_weights_h1 = xavier_init(self.n_input, self.n_hidden_recog_1)
            rec_biases_h1 = tf.constant(0.1, shape=[self.n_hidden_recog_1]) 
            rec_layer_1 = self.transfer_fct(tf.matmul(self.x, rec_weights_h1) + rec_biases_h1,name = 'recognition_network_layer_1') 
            self._summary_helper(rec_layer_1)
            
            
        with tf.variable_scope('recognition_network_layer_2'): 
            rec_weights_h2 = xavier_init(self.n_hidden_recog_1, self.n_hidden_recog_2)
            rec_biases_h2 = tf.constant(0.1, shape=[self.n_hidden_recog_2])
            rec_layer_2 = self.transfer_fct(tf.matmul(rec_layer_1, rec_weights_h2)+rec_biases_h2,name = 'recognition_network_layer_2') 
            self._summary_helper(rec_layer_2) 

            
        with tf.variable_scope('recognition_network_mean'):
            weights_out_mean = xavier_init(self.n_hidden_recog_2, self.n_z)
            biases_out_mean =  bias([self.n_z])
            z_mean = tf.add(tf.matmul(rec_layer_2, weights_out_mean),biases_out_mean, name = 'recognition_network_mean')
            self._summary_helper(z_mean) 
            
        with tf.variable_scope('recognition_network_sigma'): 
            weights_out_log_sigma = xavier_init(self.n_hidden_recog_2, self.n_z)
            biases_out_log_sigma = bias([self.n_z])
            z_log_sigma_sq = tf.add(tf.matmul(rec_layer_2, weights_out_log_sigma),biases_out_log_sigma, name = 'recognition_network_sigma')
            self._summary_helper(z_log_sigma_sq)
            
        return (z_mean, z_log_sigma_sq)
        
    #解码器/生成网络(decoder network)
    def _generator_network(self):
        with tf.variable_scope('generator_network_layer_1'):
            gen_weights_h1 = xavier_init(self.n_z, self.n_hidden_gener_1)
            gen_biases_h1 = bias([self.n_hidden_gener_1])
            gen_layer_1 = self.transfer_fct(tf.matmul(self.z, gen_weights_h1) + gen_biases_h1, name = 'generator_network_layer_1') 
            self._summary_helper(gen_layer_1)                                                
        with tf.variable_scope('generator_network_layer_2'):  
            gen_weights_h2 = xavier_init(self.n_hidden_gener_1, self.n_hidden_gener_2)
            gen_biases_h2 = bias([self.n_hidden_gener_2])        
            gen_layer_2 = self.transfer_fct(tf.matmul(gen_layer_1, gen_weights_h2) + gen_biases_h2, name = 'generator_network_layer_1') 
            self._summary_helper(gen_layer_2)                                   
        with tf.variable_scope('recognition_network_x'): 
            gen_weights_out = xavier_init(self.n_hidden_gener_2, self.n_input)
            gen_biases_out = bias([self.n_input]) 
            x_reconstr_mean = tf.nn.sigmoid(tf.matmul(gen_layer_2, gen_weights_out) + gen_biases_out, name = 'recognition_network_x')
            self._summary_helper(x_reconstr_mean)                         
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # 损失函数由两部分组成:

        with tf.variable_scope('loss'):         
            reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                               + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)

            latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                               - tf.square(self.z_mean) 
                                               - tf.exp(self.z_log_sigma_sq), 1)
            
            self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
            tf.summary.scalar('cost',self.cost)
            # Use ADAM optimizer
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def _summary_helper(self, variable):
        tf.summary.histogram(variable.op.name + '/activation', variable)
        tf.summary.scalar(variable.op.name + '/mean', tf.reduce_mean(variable))
 




 
##################################################################################数据下载函数         
def read_image(file_queue):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)
    return image, label

def read_image_batch(train_file_path, batch_size):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file_path))
    img, label = read_image(filename_queue)
    capacity = 1000 + 3 * batch_size
    image_batch = tf.train.batch([img], batch_size=batch_size, capacity=capacity, num_threads=10)
    #one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))
    return image_batch
    

    
    
    
    
    
    
##############################################################训练    
    
train_images = read_image_batch(train_file_path, FLAGS.batch_size)    

vae = VariationalAutoencoder(train_images,
                             learning_rate=FLAGS.learning_rate, 
                             batch_size=FLAGS.batch_size)
        # 创建自编码器网络
print("创建变分自编码器网络")
vae._create_network()

# 损失函数和优化器
print("损失函数和优化器")
vae._create_loss_optimizer()


sess = tf.InteractiveSession()
summary = tf.summary.FileWriter(FLAGS.summaryDir, graph=sess.graph)
saver = tf.train.Saver(var_list=tf.trainable_variables())
tf.global_variables_initializer().run() 
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess,coord = coord)                            


try:
    for epoch in range(FLAGS.training_epochs):
        if coord.should_stop():
            break
        _,cost = sess.run(fetches = [vae.optimizer,vae.cost])
        
        if (epoch*FLAGS.batch_size) % 55000 == 0 or epoch == FLAGS.training_epochs - 1:
            summary_data = tf.summary.merge_all()
            summary.add_summary(sess.run(summary_data), epoch)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            print("Step:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(cost))
        if (epoch + 1) == FLAGS.training_epochs :
            saver.save(sess=sess, save_path=os.path.join(FLAGS.checkpointDir, 'vae.model'))
except tf.errors.OutOfRangeError:
    print("Done:")
finally:
    coord.request_stop()
    
    
coord.join(threads)
summary.close()
sess.close()
print("训练成功")

