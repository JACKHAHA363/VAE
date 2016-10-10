import keras
import vae_keras
from vae_keras import VariationalAutoencoder
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

input_dim = 84*84
y_dim = 100
z_dim = 100
model = VariationalAutoencoder(input_dim, y_dim, z_dim)
model.vae.load_weights("test/model.h5")

with open("MNIST_data/translated.dat", "rb") as f:
    data = pickle.load(f)[0:100]

# test reconstruction
#reconst = model.vae.predict(data)
#layout = []
#for i in range(10):
#    row = reconst[i*10 : i*10 + 10].reshape(10, 84, 84)
#    row = np.concatenate(row, axis=1)
#    layout.append(row)
#layout = np.concatenate(layout)
#plt.imsave("test/reconst.png", layout)


# test variation disentangling
img1 = data[0]
img2 = data[1]

[z_mean1, z_log_var1, y_1] = model.enc.predict(img1.reshape(1, input_dim))
[z_mean2, z_log_var2, y_2] = model.enc.predict(img2.reshape(1, input_dim))

num_generate = 10
eps = np.random.normal(0, 1, [num_generate, z_dim])
z1 = z_mean1 + np.exp(z_log_var1 / 2) * eps
y1 = np.repeat(y_1, num_generate, axis=0)
z2 = z_mean2 + np.exp(z_log_var2 / 2) * eps
y2 = np.repeat(y_2, num_generate, axis=0)

newinput1 = [z2, y1]
newinput2 = [z1, y2]
imgs1 = model.dec.predict(newinput1)
imgs2 = model.dec.predict(newinput2)

row1 = list(imgs1.reshape(10, 84, 84))
row1.append(img1.reshape(84,84))
row1 = np.concatenate(np.array(row1), axis=1)
row2 = list(imgs2.reshape(10, 84, 84))
row2.append(img2.reshape(84,84))
row2 = np.concatenate(np.array(row2), axis=1)
layout = np.concatenate([row1, row2])
plt.imsave("test/swap_z.png", layout)

imgs1 = model.dec.predict([eps, y1])
imgs2 = model.dec.predict([eps, y2])
row1 = list(imgs1.reshape(10, 84, 84))
row1.append(img1.reshape(84,84))
row1 = np.concatenate(np.array(row1), axis=1)
row2 = list(imgs2.reshape(10, 84, 84))
row2.append(img2.reshape(84,84))
row2 = np.concatenate(np.array(row2), axis=1)
layout = np.concatenate([row1, row2])
plt.imsave("test/same_y_random_z.png", layout)

y_rand = np.zeros([10,100])
for i in range(10):
    y_rand[i][np.random.randint(100)] = 1
imgs1 = model.dec.predict([z1, y_rand])
imgs2 = model.dec.predict([z2, y_rand])
row1 = list(imgs1.reshape(10, 84, 84))
row1.append(img1.reshape(84,84))
row1 = np.concatenate(np.array(row1), axis=1)
row2 = list(imgs2.reshape(10, 84, 84))
row2.append(img2.reshape(84,84))
row2 = np.concatenate(np.array(row2), axis=1)
layout = np.concatenate([row1, row2])
plt.imsave("test/same_z_random_y.png", layout)


