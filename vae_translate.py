import keras.models
from keras.layers import Input, Dense, Lambda, Merge, merge
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

import tensorflow as tf

class VariationalAutoencoder(object):
    def __init__(sf, input_dim, y_dim, z_dim):
        # copy and paste
        sf.input_dim = input_dim
        sf.y_dim = y_dim
        sf.z_dim = z_dim

        # encoder
        sf.x = Input(shape=(input_dim,))
        sf.enc_h_1 = Dense(500, activation='tanh', input_dim=input_dim)(sf.x)
        sf.enc_h_2 = Dense(200, activation='tanh')(sf.enc_h_1)
        sf.z_mean = Dense(z_dim)(sf.enc_h_2)
        sf.z_log_var = Dense(z_dim)(sf.enc_h_2)
        sf.y_probs = Dense(y_dim, activation='softmax')(sf.enc_h_2)
        sf.enc = Model(input=sf.x, output=[sf.z_mean, sf.z_log_var, sf.y_probs])
        
        # sampling using reparameterization
        
        def sampling(args):
            mean, log_var = args
            epsilon = K.random_normal(shape=(z_dim,), mean=0, std=1)
            return mean + K.exp(log_var / 2) * epsilon
        
        sf.z = Lambda(function=sampling)([sf.z_mean, sf.z_log_var])
        
        # decoder creating layers to be reused
        z_fc = Dense(200, activation='tanh', input_dim=z_dim) 
        y_fc = Dense(200, activation='tanh', input_dim=y_dim)
        merge_layer = Merge([Sequential([z_fc]), Sequential([y_fc])], mode="concat", concat_axis=1)
        h_fc = Dense(1000, activation='tanh')
        dec_fc = Dense(input_dim, activation='sigmoid')
        sf.dec = Sequential([merge_layer, h_fc, dec_fc])
        
        sf.z_h = z_fc(sf.z)
        sf.y_h = y_fc(sf.y_probs)
        sf.merged = merge([sf.z_h, sf.y_h], mode='concat', concat_axis=1)
        sf.dec_h =h_fc(sf.merged)
        sf.x_dec = dec_fc(sf.dec_h)
        
        # total model
        sf.vae = Model(input=sf.x, output=sf.x_dec)

        ''' Use a uniform for y_prior ''' 
        sf.xent_loss = tf.reduce_mean(sf.input_dim * objectives.binary_crossentropy(sf.x, sf.x_dec))
        sf.z_loss = - tf.reduce_mean(0.5 * K.sum(1 + sf.z_log_var - K.square(sf.z_mean) - K.exp(sf.z_log_var), axis=-1))
        # omit the constant term
        sf.y_loss = tf.reduce_mean(10*K.sum(sf.y_probs * K.log(sf.y_probs * sf.y_dim), axis=-1))
        sf.loss = sf.xent_loss + sf.z_loss + sf.y_loss


