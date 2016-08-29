from vae import VariationalAutoencoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

import os
import numpy as np
#np.random.seed(0)
#tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

network_architecture = dict(
        n_hidden_recog_1=500,
        n_hidden_recog_2=500, 
        n_hidden_gener_1=500, 
        n_hidden_gener_2=500, 
        n_input=784, 
        n_z=20
        )

learning_rate=0.001
batch_size=100
training_epochs=1
display_step=5

def NoiseContrastiveLoss(data_model, noise_model):
    pos_term = tf.log(
            1.0 + tf.exp(noise_model.ll_pos - data_model.ll_pos)
        )

    neg_term = tf.log(
            1.0 + tf.exp(data_model.ll_neg - noise_model.ll_neg)
        )
    return tf.reduce_mean(pos_term + neg_term), tf.reduce_mean(noise_model.ll_pos - data_model.ll_pos), tf.reduce_mean(data_model.ll_neg - noise_model.ll_neg)


S = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

with tf.device("/gpu:0"):
    neg_example = tf.placeholder(tf.float32, [batch_size, network_architecture["n_input"]])
    pos_example = tf.placeholder(tf.float32, [batch_size, network_architecture["n_input"]])
    vae_data = VariationalAutoencoder(
            network_architecture=network_architecture, 
            transfer_fct=tf.nn.softplus,
            batch_size=batch_size,
            session=S, 
            trainable=True,
            pos_example=pos_example, 
            neg_example=neg_example
            )
    
    vae_noise = VariationalAutoencoder(
            network_architecture=network_architecture, 
            transfer_fct=tf.nn.softplus,
            batch_size=batch_size,
            session=S, 
            trainable=False,
            pos_example=pos_example, 
            neg_example=neg_example
            )
        
    loss, pos_diff, neg_diff = NoiseContrastiveLoss(vae_data, vae_noise)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    init = tf.initialize_all_variables()
    S.run(init)
 
    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(n_samples / batch_size)

        for i in range(3):
            batch_data, _ = mnist.train.next_batch(batch_size)
            batch_noise = vae_noise.reconstruct(batch_data)
            
            loss_eval, pos_eval, neg_eval = S.run(
                    [loss, pos_diff, neg_diff], 
                    {pos_example : batch_data, neg_example : batch_noise}
                    )
            print("loss: " + str(loss_eval))
            print("pos_diff: " + str(pos_eval))
            print("neg_diff: " + str(neg_eval))
#            
#            avg_loss += loss_eval / n_samples * batch_size
#            
#            print(loss_eval)

#         if epoch % display_step == 0:
#            print("Epoch:", '%04d' % (epoch+1), "nc loss =", "{:.9f}".format(avg_loss))



