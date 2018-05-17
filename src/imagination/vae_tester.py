'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
from scipy.stats import norm
import random
import random

from keras.models import load_model


batch_size = 50
latent_dim = 3
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
     b_imgs, g_imgs, b_only, g_only, b_hand, g_hand = [], [], [], [], [], []
     VAL = 0
     val = 'validation/'
     if not VAL:
         val = ''  # change to empty string to test on training set
     for filename in os.listdir(val + 'b'):
         img = cv2.imread(os.path.join(val + 'b', filename))
         if img is not None:
             b_imgs.append(processImage(img))
     for filename in os.listdir(val + 'g'):
         img = cv2.imread(os.path.join(val + 'g', filename))
         if img is not None:
             g_imgs.append(processImage(img))
     for filename in os.listdir(val + 'b_only'):
         img = cv2.imread(os.path.join(val + 'b_only', filename))
         if img is not None:
             b_only.append(processImage(img))
     for filename in os.listdir(val + 'g_only'):
         img = cv2.imread(os.path.join(val + 'g_only', filename))
         if img is not None:
             g_only.append(processImage(img))
     for filename in os.listdir(val + 'b_hand'):
         img = cv2.imread(os.path.join(val + 'b_hand', filename))
         if img is not None:
             b_hand.append(processImage(img))
     for filename in os.listdir(val + 'g_hand'):
         img = cv2.imread(os.path.join(val + 'g_hand', filename))
         if img is not None:
             g_hand.append(processImage(img))
     return b_imgs, g_imgs, b_only, g_only, b_hand, g_hand

decoder = load_model('models/decoder_12.h5')
encoder = load_model('models/encoder_12.h5')

imgs_list = []
for i in getImages():
    imgs_list.append(i)

b_imgs, g_imgs, b_only, g_only, b_hand, g_hand = getImages()
imgs = np.asarray(b_imgs +g_imgs + b_only + g_only + b_hand + g_hand)
#imgs = np.asarray(imgs_list)

im_size = 64


# draw scatter plot of image classes in latent space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, imgs_ in enumerate(imgs_list):
    encoded = encoder.predict(np.asarray(imgs_))
    encoded = np.asarray(encoded)
    #print(encoded.shape)
    ax.scatter(encoded[0, :, 0], encoded[0, :, 1], encoded[0, :, 2])
plt.show()

while True:
    c = random.choice(range(0, imgs.shape[0]))
    img = imgs[c, :, :]
    print(c)
    #img = imgs[117]
    img = img.reshape((1, 64, 64, 3))
    encoded = np.asarray(encoder.predict(img))
    encoded_logvar = encoded[1, :, :]    #store log(var) vector for later
    encoded = encoded[0, :, :]   #get just means
    print('means = ', encoded)
    print('std. dev = ', np.sqrt(np.exp(encoded_logvar)))
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
