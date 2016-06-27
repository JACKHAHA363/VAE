import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

np.random.seed(0)
tf.set_random_seed(0)

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""

    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)


class Vae_yz_x(object):
    
    def __init__(self, n_x, n_y, n_z, network_config, )
