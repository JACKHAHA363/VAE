# A vae_yz_x example for ssl for mnist. The network
# architecture is the same as original paper

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

np.random.seed(0)
tf.set_random_seed(0)

INPUT_SIZE = 784

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""

    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class Vae_yz_x(object):
    
    def __init__(self, config):
        self.config = config
        self.initialize_weights()
        self.build_model()

    def build_model(self):
        """ hold input/output data """
        self.x = tf.placeholder(tf.float32, shape=(self.config["batch_size"],784))
        self.y = tf.placeholder(tf.float32, shape=(self.config["batch_size"],10))
    
    def initialize_weights(self):
        """ parameters for the recognition_network """
        self.phi = dict(
                weight_x = tf.Variable(xavier_init(784, self.config["n_hidden_recog"]), name="recog_w_x"),
                weight_y = tf.Variable(xavier_init(10, self.config["n_hidden_recog"]), name="recog_w_y"),
                recog_b = tf.Variable(tf.zeros([self.config["n_hidden_recog"]]), name="recog_b")
                )

        """ parameters for the generative_network """
        self.theta = dict(
                
                )
    
    def z_recognition_network(self):
        
        z_mean = tf.softplus(
                tf.add(
                    self.phi["recog_b"], 
                    tf.add(
                        tf.matmul(self.x, self.phi["weight_x"]), 
                        tf.matmul(self.y, self.phi["weight_y"])
                        )
                    )
                )
        
        

        
