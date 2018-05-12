'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.stats import norm
import random
import random

from keras.layers import Input, Lambda, Conv2D, Dense, Flatten, Conv2DTranspose, Reshape
from keras.models import Model, load_model
from keras.optimizers import *
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 50
latent_dim = 4
intermediate_dim = 16
epochs = 50
epsilon_std = 1.0

IMAGE_WIDTH =64
IMAGE_HEIGHT = 64
CHANNELS = 3
original_dim = IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS


def processImage( img ):
     image = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
     image = image.astype(float)/255
     return image


def getImages():
     imgs= []
     for filename in os.listdir('b'):
         img = cv2.imread(os.path.join('b', filename))
         if img is not None:
             imgs.append(processImage(img))
     for filename in os.listdir('g'):
         img = cv2.imread(os.path.join('g', filename))
         if img is not None:
             imgs.append(processImage(img))
     for filename in os.listdir('b_only'):
         img = cv2.imread(os.path.join('b_only', filename))
         if img is not None:
             imgs.append(processImage(img))
     for filename in os.listdir('g_only'):
         img = cv2.imread(os.path.join('g_only', filename))
         if img is not None:
             imgs.append(processImage(img))
     for filename in os.listdir('b_hand'):
         img = cv2.imread(os.path.join('b_hand', filename))
         if img is not None:
             imgs.append(processImage(img))
     for filename in os.listdir('g_hand'):
         img = cv2.imread(os.path.join('g_hand', filename))
         if img is not None:
             imgs.append(processImage(img))
     return np.asarray(imgs)

decoder = load_model('models/decoder_11.h5')
encoder = load_model('models/encoder_11.h5')

imgs = getImages()
im_size = 64

while True:
    c = random.choice(range(0,imgs.shape[0]))
    img = imgs[c, :, :]
    print(c)
    #img = imgs[117]
    img = img.reshape((1,64,64,3))
    encoded = encoder.predict(img)
    decoded = decoder.predict(encoded)
    b, g, r = cv2.split(decoded.reshape(64, 64, 3))
    decoded = cv2.merge([r, g, b])
    b, g, r = cv2.split(img.reshape(64,64,3))
    img = cv2.merge([r, g, b])
#    plt.imshow(img)
#    plt.show()
    n = 10
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    #print(np.max(grid_x))
    figure = np.zeros((im_size*latent_dim, im_size * (n + 2), 3))
    for i in range(latent_dim):
        figure[im_size*i:im_size*(i+1), 0:im_size, :] = img
        figure[im_size*i:im_size*(i+1), im_size:im_size*2, :] = decoded
        for j, xj in enumerate(grid_x):
            #print(encoded[0, i])
            encoded_=np.copy(encoded)
            encoded_[0,i] = xj
            decoded_ = decoder.predict(encoded_)
            b, g, r = cv2.split(decoded_.reshape(64, 64, 3))
            decoded_ = cv2.merge([r, g, b])
            figure[im_size*i:im_size*(i+1), im_size*(j+2):im_size*(j+3), :] = decoded_
    plt.figure()
    plt.imshow(figure)
    plt.show()

    #print(encoded)
    #print(decoded.shape)
    #decoded = vae.predict(img)
    #b, g, r = cv2.split(decoded.reshape(64,64,3))
    #decoded = cv2.merge([r, g, b])
    #print(decoded)
