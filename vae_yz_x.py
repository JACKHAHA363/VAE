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

config = dict(
        batch_size=100,
        config_z_xy=[500,500], # architecture for the q(z|x,y) network
        config_x_yz=[500,500], # architecture for the p(x|y,z) network
        n_z=50, # dimension of z
        nonlinear_q=tf.softplus, # the non-linear function each layer in all q network
        nonlinear_p=tf.softplus, # the non-linear function each layer in all p network
        )

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
        """ parameters for the recognition_network of q(z|x,y)"""
        config_z_xy = self.config["config_z_xy"]
        phi = dict(
                wx = tf.Variable(xavier_init(784, config_z_xy[0]), name="wx"),
                wy = tf.Variable(xavier_init(10, config_z_xy[0]), name="wy"),
                b0 = tf.Variable(tf.zeros([config_z_xy[0]]), name="b0")
                )
        # initialize weight for each level
        for i in range(1, len(config_z_xy)):
            phi["w"+str(i)] = tf.Variable(xavier_init(config_z_xy[i-1], config_z_xy[i]), name="w"+str(i))
            phi["b"+str(i)] = tf.Variable(tf.zeros([config_z_xy[i]]), name="b"+str(i))
        
        # initialize weight for statistical parameter. Here we choose Gaussian
        phi["w_mean"] = tf.Variable(xavier_init(config_z_xy[-1], self.config["n_z"]), name="w_mean")
        phi["b_mean"] = tf.Variable(tf.zeros([self.config["n_z"]]), name="b_mean")
        phi["w_logvar"] = tf.Variable(xavier_init(config_z_xy[-1], self.config["n_z"]), name="w_logvar")
        phi["b_logvar"] = tf.Variable(tf.zeros([self.config["n_z"]]), name="b_logvar")

        """ parameters for the generative_network of p(x|y,z) """
        config_x_yz = self.config["config_x_yz"]
        theta = dict(
                wy = tf.Variable(xavier_init(10, config_x_yz[0]), name="wy")
                wz = tf.Variable(xavier_init(self.config["n_z"], config_x_yz[0]), name="wz")
                b0 = tf.Variable(tf.zeros([config_x_yz[0]]), name="b0")
                )
        
        for i in range(1, len(config_x_yz)):
            theta["w"+str(i)] = tf.Variable(xavier_init(config_x_yz[i-1], config_x_yz[i]), name="w"+str(i))
            theta["b"+str(i)] = tf.Variable(tf.zeros([config_x_yz[i]]), name="b"+str(i))
        
        # Also use gaussian
        theta["w_mean"] = tf.Variable(xavier_init(config_x_yz[-1], 784), name="w_mean")
        theta["b_mean"] = tf.Variable(tf.zeros([784]), name="b_mean")
        theta["w_logvar"] = tf.Variable(xavier_init(config_x_yz[-1], 784), name="w_logvar")
        theta["b_logvar"] = tf.Variable(tf.zeros([784]), name="b_logvar")

        self.phi = phi
        self.theta = theta
    
    def z_recognition_network(self):
        
        

        
