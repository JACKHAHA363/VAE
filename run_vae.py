from vae_keras import VariationalAutoencoder
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# load translated data
import pickle
with open("MNIST_data/translated.dat", "rb") as f:
    data = pickle.load(f)
train_data = data[0 : int(0.8 * len(data))]
valid_data = data[int(0.8 * len(data)) :]

input_dim = 84*84
y_dim = 100
z_dim = 100
model = VariationalAutoencoder(input_dim, y_dim, z_dim)
model.vae.compile(optimizer='rmsprop', loss=model.vae_loss)

hist = model.vae.fit(train_data, train_data,
        shuffle=True,
        nb_epoch=500,
        batch_size=500,
        validation_data=(valid_data, valid_data),
        )
loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.figure()
plt.title("loss")
plt.plot(loss)
plt.savefig("test/training_loss")

plt.figure()
plt.title("val loss")
plt.plot(val_loss)
plt.savefig("test/validation_loss")

model.vae.save_weights("test/model.h5")
