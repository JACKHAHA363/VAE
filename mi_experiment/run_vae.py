# This script uses a vinilla VAE to check if
# mutual information can be optimized by optimizing
# KL lower bound
from vae_mi import VariationalAutoencoder
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# load translated data
import pickle
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping
import keras.backend.tensorflow_backend as T
from keras.datasets import mnist

(train, _), (test, _) = mnist.load_data()

train_data = np.array(train).reshape([-1,28*28]).astype(np.float32) / 255
valid_data = np.array(test).reshape([-1,28*28]).astype(np.float32) / 255

input_dim = 28*28
z_dim = 10

early_stop = EarlyStopping(monitor='val_loss', patience=20, mode="min")

class CalculateMI(Callback):
    # This call back function caculate variationl lower bound for mutual information of z and x
    def __init__(self, sample_num, dec, enc):
        super(CalculateMI, self).__init__()
        
        self.dec = dec
        self.enc = enc
        self.sample_num = sample_num

    def on_epoch_end(self, epoch, logs):
        z = np.random.normal(0, 1, size=[self.sample_num, z_dim]).astype(np.float32)
        x_gen = self.dec.predict(z)
        z_mean, z_log_var = self.enc.predict(x_gen)
        result = - z_log_var - np.square(z-z_mean) / np.exp(z_log_var)
        logs["avg_mi"] = np.mean(np.sum(result, axis=-1))



model = VariationalAutoencoder(input_dim, z_dim)
calcmi = CalculateMI(sample_num=500, dec=model.dec, enc=model.enc)

model.vae.compile(optimizer='rmsprop', loss=model.vae_loss)

hist = model.vae.fit(train_data, train_data,
        shuffle=True,
        nb_epoch=500,
        batch_size=100,
        validation_data=(valid_data, valid_data),
        callbacks=[calcmi],
        )

loss = hist.history['loss']
val_loss = hist.history['val_loss']
avg_mi = hist.history['avg_mi']

plt.figure()
plt.title("loss")
plt.plot(loss)
plt.savefig("test/training_loss")

plt.figure()
plt.title("val loss")
plt.plot(val_loss)
plt.savefig("test/validation_loss")

plt.figure()
plt.title("avg_mi")
plt.plot(avg_mi)
plt.savefig("test/avg_mi")

model.vae.save_weights("test/model.h5")
