import keras
import vae_keras
from vae_keras import VariationalAutoencoder
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_dim = 84*84
y_dim = 100
z_dim = 100
model = VariationalAutoencoder(input_dim, y_dim, z_dim)
model.vae.load_weights("test/model.h5")

with open("MNIST_data/translated.dat", "rb") as f:
    data = pickle.load(f)[0:100]
reconst = model.vae.predict(data)

layout = []
for i in range(10):
    row = reconst[i*10 : i*10 + 10].reshape(10, 84, 84)
    row = np.concatenate(row, axis=1)
    layout.append(row)
layout = np.concatenate(layout)
plt.imsave("test/reconst.png", layout)


