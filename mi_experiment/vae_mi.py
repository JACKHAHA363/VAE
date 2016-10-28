import keras.models
from keras.layers import Input, Dense, Lambda, Merge, merge
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

class VariationalAutoencoder(object):
    def __init__(sf, input_dim, z_dim):
        # copy and paste
        sf.input_dim = input_dim
        sf.z_dim = z_dim

        # encoder
        sf.x = Input(shape=(input_dim,))
        sf.enc_h = Dense(200, activation='tanh')(sf.x)
        sf.z_mean = Dense(z_dim)(sf.enc_h)
        sf.z_log_var = Dense(z_dim)(sf.enc_h)
        sf.enc = Model(input=sf.x, output=[sf.z_mean, sf.z_log_var])
        
        # sampling using reparameterization
        
        def sampling(args):
            mean, log_var = args
            epsilon = K.random_normal(shape=(z_dim,), mean=0, std=1)
            return mean + K.exp(log_var / 2) * epsilon
        
        sf.z = Lambda(function=sampling)([sf.z_mean, sf.z_log_var])
        
        # decoder creating layers to be reused
        z_fc = Dense(200, activation='tanh', input_dim=z_dim) 
        dec_fc = Dense(input_dim, activation='sigmoid')
        sf.dec = Sequential([z_fc, dec_fc])
        
        sf.z_h = z_fc(sf.z)
        sf.x_dec = dec_fc(sf.z_h)
        
        # total model
        sf.vae = Model(input=sf.x, output=sf.x_dec)

    def vae_loss(sf, x, x_dec):
        ''' Use a uniform for y_prior ''' 
        xent_loss = sf.input_dim * objectives.binary_crossentropy(x, x_dec)
        z_loss = - 0.5 * K.sum(1 + sf.z_log_var - K.square(sf.z_mean) - K.exp(sf.z_log_var), axis=-1)
        return xent_loss + z_loss 


