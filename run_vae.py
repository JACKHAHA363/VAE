from vae import VariationalAutoencoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

import os
import numpy as np
#np.random.seed(0)
#tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

network_architecture = dict(
        n_hidden_recog_1=50,
        n_hidden_recog_2=50, 
        n_hidden_gener_1=50, 
        n_hidden_gener_2=50, 
        n_input=784, 
        n_z=10
        )

learning_rate=0.001
batch_size=100
training_epochs=95

with tf.device("/gpu:1"):
    neg_example = tf.placeholder(tf.float32, [batch_size, network_architecture["n_input"]])
    pos_example = tf.placeholder(tf.float32, [batch_size, network_architecture["n_input"]])
    S = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    vae_data = VariationalAutoencoder(
            network_architecture=network_architecture, 
            transfer_fct=tf.nn.softplus,
            batch_size=batch_size,
            session=S, 
            trainable=True,
            pos_example=pos_example, 
            neg_example=neg_example,
            weight_init=None
            )
    
    LB = []
    loss = -tf.reduce_sum(vae_data.lb_pos) / batch_size
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
               
    init = tf.initialize_all_variables()
    S.run(init)
    
    for epoch in range(training_epochs):
        
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            batch_data, _ = mnist.train.next_batch(batch_size)
            
            loss_eval, _ = S.run(
                    [loss, opt], 
                    {pos_example : batch_data}
                    )
            
            LB.append(-loss_eval)

        print("epoch " + str(epoch+1) + ": " + str(-loss_eval))

learned_weights = vae_data.get_weight_init()
with open("normal/model.dat", "wb") as f:
    pickle.dump(learned_weights, f)   
plt.figure()
plt.plot(LB)
plt.savefig("normal/evaluation.png")

samples = vae_data.sample()
layout = []
for i in range(10):
    row = samples[i*10 : i*10 + 10].reshape(10, 28, 28)
    row = np.concatenate(row, axis=1)
    layout.append(row)
layout = np.concatenate(layout)
plt.imsave("normal/sample.png", layout)
