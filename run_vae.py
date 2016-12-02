import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

from vae_translate import VariationalAutoencoder as vae
import tensorflow as tf

# set limit on the memory usage on GPU
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

# some parameters
length = 28
input_dim = 784
z_dim = 1
y_dim = 10
nb_epoch = 100

model = vae(input_dim=input_dim, y_dim=y_dim, z_dim=z_dim)
train_step = tf.train.AdamOptimizer().minimize(model.loss)

# train my VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# training configuration
batch_size = 100
epochs = 10
train = []
test = []
reconst = []
zl = []
yl = []

init = tf.initialize_all_variables()
with sess.as_default():
    init.run()

with sess.as_default():
    for i in range(epochs):
        for batch_idx in range(0, len(x_train), batch_size):
            train_step.run(
                    feed_dict={model.x : x_train[batch_idx : batch_idx + batch_size]}
                    )
        
        train_loss = model.loss.eval(feed_dict={model.x : x_train})
        test_loss, reconst_loss, y_loss, z_loss = sess.run(
                [
                    model.loss, 
                    tf.reduce_mean(model.xent_loss),
                    tf.reduce_mean(model.y_loss), 
                    tf.reduce_mean(model.z_loss)
                    ],
                feed_dict={model.x : x_test}
                )
        print("at epoch {}, first train batch: {}, test: {}".format(i, train_loss, test_loss))
        test.append(test_loss)
        train.append(train_loss)
        reconst.append(reconst_loss)
        zl.append(z_loss)
        yl.append(y_loss)

# plot training result
plt.figure()
plt.title("train loss")
plt.plot(train)
plt.savefig("test/train_loss.png")

plt.figure()
plt.title("test loss")
plt.plot(test)
plt.savefig("test/test_loss.png")

plt.figure()
plt.title("reconst loss")
plt.plot(reconst)
plt.savefig("test/reconst_loss.png")

plt.figure()
plt.title("z loss")
plt.plot(zl)
plt.savefig("test/z_loss.png")

plt.figure()
plt.title("y loss")
plt.plot(yl)
plt.savefig("test/y_loss.png")

# reconstruction 
test_data = x_test[0:100]
reconst = model.x_dec.eval(feed_dict={model.x : test_data}, session=sess)
layout = []
for i in range(10):
    row = reconst[i*10 : i*10 + 10].reshape(10, length, length)
    row = np.concatenate(row, axis=1)
    layout.append(row)
layout = np.concatenate(layout)
plt.imsave("test/reconst.png", layout)

# swap z and y
img1 = test_data[0]
img2 = test_data[1]

[z_mean1, z_log_var1, y_1] = sess.run(
        [model.z_mean, model.z_log_var, model.y_probs],
        feed_dict={model.x : img1.reshape(1,input_dim)}
        )
[z_mean2, z_log_var2, y_2] = sess.run(
        [model.z_mean, model.z_log_var, model.y_probs],
        feed_dict={model.x : img2.reshape(1,input_dim)}
        )
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
row1 = list(imgs1.reshape(10, length, length))
row1.append(img1.reshape(length,length))
row1 = np.concatenate(np.array(row1), axis=1)
row2 = list(imgs2.reshape(10, length, length))
row2.append(img2.reshape(length,length))
row2 = np.concatenate(np.array(row2), axis=1)
layout = np.concatenate([row1, row2])
plt.imsave("test/swap_z.png", layout)

newinput1 = [z1, y2]
newinput2 = [z2, y1]
imgs1 = model.dec.predict(newinput1)
imgs2 = model.dec.predict(newinput2)
row1 = list(imgs1.reshape(10, length, length))
row1.append(img1.reshape(length, length))
row1 = np.concatenate(np.array(row1), axis=1)
row2 = list(imgs2.reshape(10, length, length))
row2.append(img2.reshape(length, length))
row2 = np.concatenate(np.array(row2), axis=1)
layout = np.concatenate([row1, row2])
plt.imsave("test/swap_y.png", layout)





