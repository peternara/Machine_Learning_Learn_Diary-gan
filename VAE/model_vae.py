import tensorflow as tf
import numpy as np
import os

np.random.seed(0)
tf.set_random_seed(0)

#xavier参数初始化         
def xavier_init(shape_in,shape_out, constant=1):
    low = -constant*np.sqrt(6.0/(shape_in + shape_out)) 
    high = constant*np.sqrt(6.0/(shape_in + shape_out))

    return tf.Variable(tf.random_uniform((shape_in, shape_out),minval = low, maxval=high,dtype=tf.float32))

def bias(shape):
    return tf.constant(0.1, shape=shape)

class VariationalAutoencoder(object):

    def __init__(self, transfer_fct=tf.nn.relu, # softplus sigmoid, softplus  ,
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
        
        # 输入占位符
        self.x =  tf.placeholder(tf.float32, [self.batch_size, 784])
        # # 创建自编码器网络
        # self._create_network()
        
        # # 损失函数和优化器
        # self._create_loss_optimizer()
        
    
    def _create_network(self):
        self.z_mean, self.z_log_sigma_sq = self._recognition_network()
        
        #重新参数化生成z
        eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float32)
        with tf.variable_scope('z'): 
            # z = mu + sigma*epsilon
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
            self._summary_helper(self.z)

        self.x_reconstr_mean = self._generator_network()
            

 
    #编码网络encode network 
    def _recognition_network(self):
        with tf.variable_scope('recognition_network_layer_1'): 
            rec_weights_h1 = xavier_init(self.n_input, self.n_hidden_recog_1)
            rec_biases_h1 =  bias([self.n_hidden_recog_1])
            tf.summary.histogram('rec_weights_h1', rec_weights_h1)
            tf.summary.histogram('rec_biases_h1', rec_biases_h1)
            rec_layer_1 = self.transfer_fct(tf.matmul(self.x, rec_weights_h1) + rec_biases_h1,name = 'recognition_network_layer_1') 
            self._summary_helper(rec_layer_1)
            
            
        with tf.variable_scope('recognition_network_layer_2'): 
            rec_weights_h2 = xavier_init(self.n_hidden_recog_1, self.n_hidden_recog_2)
            rec_biases_h2 = bias([self.n_hidden_recog_2])
            tf.summary.histogram('rec_weights_h2', rec_weights_h2)
            tf.summary.histogram('rec_biases_h2', rec_biases_h2)            
            rec_layer_2 = self.transfer_fct(tf.matmul(rec_layer_1, rec_weights_h2)+rec_biases_h2,name = 'recognition_network_layer_2') 
            self._summary_helper(rec_layer_2) 

            
        with tf.variable_scope('recognition_network_mean'):
            weights_out_mean = xavier_init(self.n_hidden_recog_2, self.n_z)
            biases_out_mean =  bias([self.n_z])
            tf.summary.histogram('weights_out_mean', weights_out_mean)
            tf.summary.histogram('biases_out_mean', biases_out_mean)            
            z_mean = tf.add(tf.matmul(rec_layer_2, weights_out_mean),biases_out_mean, name = 'recognition_network_mean')
            self._summary_helper(z_mean) 
            
        with tf.variable_scope('recognition_network_sigma'): 
            weights_out_log_sigma = xavier_init(self.n_hidden_recog_2, self.n_z)
            biases_out_log_sigma = bias([self.n_z])
            tf.summary.histogram('weights_out_log_sigma', weights_out_log_sigma)
            tf.summary.histogram('biases_out_log_sigma', biases_out_log_sigma) 
            z_log_sigma_sq = tf.add(tf.matmul(rec_layer_2, weights_out_log_sigma),biases_out_log_sigma, name = 'recognition_network_sigma')
            self._summary_helper(z_log_sigma_sq)
            
        return (z_mean, z_log_sigma_sq)
        
    #解码器/生成网络(decoder network)
    def _generator_network(self):
        with tf.variable_scope('generator_network_layer_1'):
            gen_weights_h1 = xavier_init(self.n_z, self.n_hidden_gener_1)
            gen_biases_h1 = bias([self.n_hidden_gener_1])
            tf.summary.histogram('gen_weights_h1', gen_weights_h1)
            tf.summary.histogram('gen_biases_h1', gen_biases_h1) 
            gen_layer_1 = self.transfer_fct(tf.matmul(self.z, gen_weights_h1) + gen_biases_h1, name = 'generator_network_layer_1') 
            self._summary_helper(gen_layer_1)                                                
        with tf.variable_scope('generator_network_layer_2'):  
            gen_weights_h2 = xavier_init(self.n_hidden_gener_1, self.n_hidden_gener_2)
            gen_biases_h2 = bias([self.n_hidden_gener_2]) 
            tf.summary.histogram('gen_weights_h2', gen_weights_h2)
            tf.summary.histogram('gen_biases_h2', gen_biases_h2) 
            gen_layer_2 = self.transfer_fct(tf.matmul(gen_layer_1, gen_weights_h2) + gen_biases_h2, name = 'generator_network_layer_1') 
            self._summary_helper(gen_layer_2)                                   
        with tf.variable_scope('generator_network_x'): 
            gen_weights_out = xavier_init(self.n_hidden_gener_2, self.n_input)
            gen_biases_out = bias([self.n_input])
            tf.summary.histogram('gen_weights_out', gen_weights_out)
            tf.summary.histogram('gen_biases_out', gen_biases_out) 
            x_reconstr_mean = tf.nn.sigmoid(tf.matmul(gen_layer_2, gen_weights_out) + gen_biases_out, name = 'generator_network_x')
            self._summary_helper(x_reconstr_mean)                         
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # 损失函数由两部分组成:

        with tf.variable_scope('loss'):         
            reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                               + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)

            latent_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
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