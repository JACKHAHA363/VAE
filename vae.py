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

    def __init__(self, network_architecture, transfer_fct, batch_size, session, trainable, pos_example, neg_example):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.batch_size = batch_size
        self.trainable=trainable
        self.sess = session
        self.pos_example = pos_example
        self.neg_example = neg_example

        # Create autoencoder network
        self._create_network()

        self._create_log_likelihood()
   
    def _create_network(self):

        network_weights = self._initialize_weights(**self.network_architecture)
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)

        # graph for pos_example
        self.z_mean_pos, self.z_log_sigma_sq_pos = self._recognition_network(
                network_weights["weights_recog"], 
                network_weights["biases_recog"], 
                self.pos_example)
 
        self.z_pos = tf.add(
                self.z_mean_pos, 
                tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq_pos)), eps)
                )

        self.x_reconstr_mean_pos = self._generator_network(
                network_weights["weights_gener"], 
                network_weights["biases_gener"],
                self.z_pos)
        
        # graph for neg_example
        self.z_mean_neg, self.z_log_sigma_sq_neg = self._recognition_network(
                network_weights["weights_recog"], 
                network_weights["biases_recog"], 
                self.neg_example)

        self.z_neg = tf.add(
                self.z_mean_neg, 
                tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq_neg)), eps)
                )

        self.x_reconstr_mean_neg = self._generator_network(
                network_weights["weights_gener"], 
                network_weights["biases_gener"],
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
        self.ll_pos = tf.reduce_sum(
                self.pos_example * tf.log(1e-10 + self.x_reconstr_mean_pos) + (1-self.pos_example) * tf.log(1e-10 + 1 - self.x_reconstr_mean_pos),
                1
                )
        self.ll_neg = tf.reduce_sum(
                self.neg_example * tf.log(1e-10 + self.x_reconstr_mean_neg) + (1-self.neg_example) * tf.log(1e-10 + 1 - self.x_reconstr_mean_neg),
                1
                )
      
    def reconstruct(self, X):
        samples = tf.select(
                tf.greater(
                    tf.random_uniform(tf.shape(self.pos_example)), 
                    self.x_reconstr_mean_pos
                    ),
                tf.ones_like(self.pos_example),
                tf.zeros_like(self.pos_example)
                )
        return self.sess.run(samples, {self.pos_example:X})
    
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1), trainable=self.trainable),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2), trainable=self.trainable),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z), trainable=self.trainable),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z), trainable=self.trainable)}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32), trainable=self.trainable),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32), trainable=self.trainable),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32), trainable=self.trainable),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32), trainable=self.trainable)}
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
        return all_weights
 
