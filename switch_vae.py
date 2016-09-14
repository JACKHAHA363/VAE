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
batch_size=400
training_epochs=45
converge_step=5

def NoiseContrastiveLoss(data_model, noise_model):
    pos_term = tf.log(
            1.0 + tf.exp(noise_model.lb_pos - data_model.lb_pos)
        )

    neg_term = tf.log(
            1.0 + tf.exp(data_model.lb_neg - noise_model.lb_neg)
        )

    return tf.reduce_sum(pos_term + neg_term) / batch_size


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
    
    vae_noise = VariationalAutoencoder(
            network_architecture=network_architecture, 
            transfer_fct=tf.nn.softplus,
            batch_size=batch_size,
            session=S, 
            trainable=False,
            pos_example=pos_example, 
            neg_example=neg_example,
            weight_init=None
            )
    
    training_loss = []
    LB = []

    loss = NoiseContrastiveLoss(vae_data, vae_noise)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
               
    init = tf.initialize_all_variables()
    S.run(init)
    
    for epoch in range(training_epochs):
        
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            batch_data, _ = mnist.train.next_batch(batch_size)
            batch_noise = vae_noise.sample()
            
            lb_eval, loss_eval, _ = S.run(
                    [vae_data.eval_result, loss, opt], 
                    {pos_example : batch_data, neg_example : batch_noise}
                    )
            
            training_loss.append(loss_eval)
            LB.append(lb_eval)

        print("epoch " + str(epoch+1))
        
        if (epoch+1) % converge_step == 0:
            learned_weights = vae_data.get_weight_init()
            S.close()

            # create new data and noise model
            S = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            vae_data = VariationalAutoencoder(
                network_architecture=network_architecture, 
                transfer_fct=tf.nn.softplus,
                batch_size=batch_size,
                session=S, 
                trainable=True,
                pos_example=pos_example, 
                neg_example=neg_example,
                weight_init=learned_weights,
                )
    
            vae_noise = VariationalAutoencoder(
                network_architecture=network_architecture, 
                transfer_fct=tf.nn.softplus,
                batch_size=batch_size,
                session=S, 
                trainable=False,
                pos_example=pos_example, 
                neg_example=neg_example,
                weight_init=learned_weights
                )
 
            loss = NoiseContrastiveLoss(vae_data, vae_noise)
            opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)       
            init = tf.initialize_all_variables()
            S.run(init)

plt.figure()
plt.plot(training_loss)
plt.savefig("loss.png")

plt.figure()
plt.plot(LB)
plt.savefig("evaluation.png")

batch_data, _ = mnist.train.next_batch(batch_size)
batch_recons = vae_data.reconstruct(batch_data)

plt.imsave("origin.png", batch_data[0].reshape(28,28))
plt.imsave("reconst.png", batch_recons[0].reshape(28,28))
