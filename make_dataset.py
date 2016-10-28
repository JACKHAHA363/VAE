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

all_data = np.concatenate([mnist.train.images, mnist.test.images])
data_size = len(all_data)

def combine_two_image(img1, img2):
    ''' input two (28*28) mnist image. Combine them with random translation output a (84*84) '''
    move_x1, move_y1 = np.random.randint(-28,28,size=2)
    
    canvas = np.zeros([84,84])

    # generate translation of img2, make sure not much overlap
    move_x2, move_y2 = np.random.randint(-28,28,size=2)
    while (move_x1 - move_x2)**2 + (move_y1 - move_y2)**2 <= 440:
        move_x2, move_y2 = np.random.randint(-28,28,size=2)
    
    canvas[28 - move_x1 : 56 - move_x1, 28 - move_y1 : 56 - move_y1] = img1
    canvas[28 - move_x2 : 56 - move_x2, 28 - move_y2 : 56 - move_y2] = img2

    return canvas

# create 70000 data first
translated_data = []
for i in range(70000):
    index1, index2 = np.random.randint(data_size, size=2)
    img1 = all_data[index1].reshape(28,28)
    img2 = all_data[index2].reshape(28,28)
    combined = combine_two_image(img1, img2)
    translated_data.append(combined.reshape(84*84))
    if i % 10000 == 0:
        print("finishing" + str(i) + " examples")

translated_data = np.array(translated_data)

# plot the first 100 translated_data
layout = []
samples = translated_data[0:100]
for i in range(10):
    row = samples[i*10 : i*10 + 10].reshape(10, 84, 84)
    row = np.concatenate(row, axis=1)
    layout.append(row)
layout = np.concatenate(layout)
plt.imsave("MNIST_data/translated_sample.png", layout)

import pickle
with open("MNIST_data/translated.dat", "wb") as f:
    pickle.dump(translated_data, f)

with open("MNIST_data/translated_small.dat", "wb") as f:
    pickle.dump(translated_data[0:int(len(translated_data)/10)], f)

