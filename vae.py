import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""

    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class VariationalAutoencoder(object):

    def __init__(self, network_architecture, transfer_fct, batch_size, session, trainable, pos_example, neg_example, weight_init):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.batch_size = batch_size
        self.trainable=trainable
        self.sess = session
        self.pos_example = pos_example
        self.neg_example = neg_example
        self.weight_init = weight_init

        self._create_network()

        self._create_log_likelihood()
   
    def _create_network(self):

        self.network_weights = self._initialize_weights(**self.network_architecture)
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)

        # graph for pos_example
        self.z_mean_pos, self.z_log_sigma_sq_pos = self._recognition_network(
                self.network_weights["weights_recog"], 
                self.network_weights["biases_recog"], 
                self.pos_example)
 
        self.z_pos = tf.add(
                self.z_mean_pos, 
                tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq_pos)), eps)
                )

        self.x_reconstr_mean_pos = self._generator_network(
                self.network_weights["weights_gener"], 
                self.network_weights["biases_gener"],
                self.z_pos)
        
        # graph for neg_example
        self.z_mean_neg, self.z_log_sigma_sq_neg = self._recognition_network(
                self.network_weights["weights_recog"], 
                self.network_weights["biases_recog"], 
                self.neg_example)

        self.z_neg = tf.add(
                self.z_mean_neg, 
                tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq_neg)), eps)
                )

        self.x_reconstr_mean_neg = self._generator_network(
                self.network_weights["weights_gener"], 
                self.network_weights["biases_gener"],
                self.z_neg)

    def _recognition_network(self, weights, biases, x):

        layer_1 = self.transfer_fct(tf.add(
            tf.matmul(x, weights['h1']), 
            biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(
            tf.matmul(layer_1, weights['h2']), 
            biases['b2'])) 
        z_mean = tf.add(
                tf.matmul(layer_2, weights['out_mean']), 
                biases['out_mean'])
        z_log_sigma_sq = tf.add(
                tf.matmul(layer_2, weights['out_log_sigma']), 
                biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases, z):

        layer_1 = self.transfer_fct(tf.add(
            tf.matmul(z, weights['h1']), 
            biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(
            tf.matmul(layer_1, weights['h2']), 
            biases['b2'])) 
        x_reconstr_mean = tf.nn.sigmoid(tf.add(
            tf.matmul(layer_2, weights['out_mean']), 
            biases['out_mean']))
        return x_reconstr_mean
            
    def _create_log_likelihood(self):
        self.reconst_pos = tf.reduce_sum(
                    self.pos_example * tf.log(1e-10 + self.x_reconstr_mean_pos) + (1-self.pos_example) * tf.log(1e-10 + 1 - self.x_reconstr_mean_pos),
                    1
                    )
        self.latent_pos = 0.5 * tf.reduce_sum(
                    1 + self.z_log_sigma_sq_pos - tf.square(self.z_mean_pos) - tf.exp(self.z_log_sigma_sq_pos),
                    1
                )
        
        self.lb_pos = self.reconst_pos + self.latent_pos
        
        self.reconst_neg = tf.reduce_sum(
                    self.neg_example * tf.log(1e-10 + self.x_reconstr_mean_neg) + (1-self.neg_example) * tf.log(1e-10 + 1 - self.x_reconstr_mean_neg),
                    1
                    )
        self.latent_neg = 0.5 * tf.reduce_sum(
                    1 + self.z_log_sigma_sq_neg - tf.square(self.z_mean_neg) - tf.exp(self.z_log_sigma_sq_neg),
                    1
                )
 
        self.lb_neg = self.reconst_neg - self.latent_neg
        
        self.eval_result = tf.reduce_sum(self.lb_pos) / self.batch_size

        
            
      
    def reconstruct(self, X):
        
        return self.sess.run(self.x_reconstr_mean_pos, {self.pos_example:X})
    def sample(self, z_mu=None):
        
        if z_mu is None:
            z_mu = np.random.normal(size=(self.batch_size, self.network_architecture["n_z"]))

        return self.sess.run(self.x_reconstr_mean_pos, {self.z_pos : z_mu})


    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        if self.weight_init is None:
            all_weights['weights_recog'] = {
                'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1), trainable=self.trainable),
                'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2), trainable=self.trainable),
                'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z), trainable=self.trainable),
                'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z), trainable=self.trainable)
                }
            all_weights['biases_recog'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32), trainable=self.trainable),
                'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32), trainable=self.trainable),
                'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32), trainable=self.trainable),
                'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32), trainable=self.trainable)
                }
            all_weights['weights_gener'] = {
                'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1), trainable=self.trainable),
                'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2), trainable=self.trainable),
                'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input), trainable=self.trainable),
                }
            all_weights['biases_gener'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32), trainable=self.trainable),
                'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32), trainable=self.trainable),
                'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32), trainable=self.trainable),
                }
        else:
            all_weights['weights_recog'] = {
                'h1': tf.Variable(self.weight_init['weights_recog']['h1'], trainable=self.trainable),
                'h2': tf.Variable(self.weight_init['weights_recog']['h2'], trainable=self.trainable),
                'out_mean': tf.Variable(self.weight_init['weights_recog']['out_mean'], trainable=self.trainable),
                'out_log_sigma': tf.Variable(self.weight_init['weights_recog']['out_log_sigma'], trainable=self.trainable)
                }
            all_weights['biases_recog'] = {
                'b1': tf.Variable(self.weight_init['biases_recog']['b1'], trainable=self.trainable),
                'b2': tf.Variable(self.weight_init['biases_recog']['b2'], trainable=self.trainable),
                'out_mean': tf.Variable(self.weight_init['biases_recog']['out_mean'], trainable=self.trainable),
                'out_log_sigma': tf.Variable(self.weight_init['biases_recog']['out_log_sigma'], trainable=self.trainable)
                }
            all_weights['weights_gener'] = {
                'h1': tf.Variable(self.weight_init['weights_gener']['h1'], trainable=self.trainable),
                'h2': tf.Variable(self.weight_init['weights_gener']['h2'], trainable=self.trainable),
                'out_mean': tf.Variable(self.weight_init['weights_gener']['out_mean'], trainable=self.trainable),
                }
            all_weights['biases_gener'] = {
                'b1': tf.Variable(self.weight_init['biases_gener']['b1'], trainable=self.trainable),
                'b2': tf.Variable(self.weight_init['biases_gener']['b2'], trainable=self.trainable),
                'out_mean': tf.Variable(self.weight_init['biases_gener']['out_mean'], trainable=self.trainable),
                }
 
        return all_weights

    def get_weight_init(self):
        weight_init = dict()
        weight_init['weights_recog'] = {
            'h1': self.sess.run(self.network_weights['weights_recog']['h1']),
            'h2': self.sess.run(self.network_weights['weights_recog']['h2']),
            'out_mean': self.sess.run(self.network_weights['weights_recog']['out_mean']),
            'out_log_sigma': self.sess.run(self.network_weights['weights_recog']['out_log_sigma'])
            }
        weight_init['biases_recog'] = {
            'b1': self.sess.run(self.network_weights['biases_recog']['b1']),
            'b2': self.sess.run(self.network_weights['biases_recog']['b2']),
            'out_mean': self.sess.run(self.network_weights['biases_recog']['out_mean']),
            'out_log_sigma': self.sess.run(self.network_weights['biases_recog']['out_log_sigma']),
            }
        weight_init['weights_gener'] = {
            'h1': self.sess.run(self.network_weights['weights_gener']['h1']),
            'h2': self.sess.run(self.network_weights['weights_gener']['h2']),
            'out_mean': self.sess.run(self.network_weights['weights_gener']['out_mean']),
            }
        weight_init['biases_gener'] = {
            'b1': self.sess.run(self.network_weights['biases_gener']['b1']),
            'b2': self.sess.run(self.network_weights['biases_gener']['b2']),
            'out_mean': self.sess.run(self.network_weights['biases_gener']['out_mean']),
            }
        return weight_init
   
