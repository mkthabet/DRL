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

from keras.layers import Input, Lambda, Conv2D, Dense, Flatten, Conv2DTranspose, Reshape
from keras.models import Model
from keras.optimizers import *
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 30
latent_dim = 3
intermediate_dim = 8
epochs = 50
epsilon_std = 1.0

IMAGE_WIDTH =64
IMAGE_HEIGHT = 64
CHANNELS = 3
original_dim = IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS


def processImage( img ):
    #rgb = None
    image = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.astype(float)/255
    #print(np.max(image))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #i = plt.imshow(image)
    #plt.show()
    return image
    #r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance
    #o = gray.astype('float32') / 128 - 1    # normalize
    #return o

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

def vae_loss(x, x_decoded_mean):
    x= K.batch_flatten(x)
    x_decoded_mean = K.batch_flatten(x_decoded_mean)
    xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
    #print ('shape = ', K.shape(metrics.binary_crossentropy(x, x_decoded_mean)))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #print('shape = ', kl_loss)
    return K.mean(xent_loss + kl_loss)
    #return K.mean((xent_loss)+ kl_loss)


img_input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', data_format='channels_last')(img_input)
conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
conv = Conv2D(128, (4, 4), strides=(2, 2), activation='relu')(conv)
conv2layer = Conv2D(256, (4, 4), activation='relu')
conv_out = conv2layer(conv)
conv_out_layer = Flatten(name='flatten')
h = conv_out_layer(conv_out)
z_mean = Dense(units=latent_dim)(h)
z_log_var = Dense(units=latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

print(conv2layer.output_shape[1:])


decoder_input = Input(shape=(latent_dim,))
h_decoded = Dense(units=1024, activation='relu')(decoder_input)
h_decoded = Reshape((1, 1, 1024))(h_decoded)
deconv = Conv2DTranspose(128, (5, 5), strides= (2,2), activation='relu')(h_decoded)
deconv = Conv2DTranspose(64, (5, 5), strides= (2,2), activation='relu')(deconv)
deconv = Conv2DTranspose(32, (6, 6), strides= (2,2), activation='relu')(deconv)
decoded_mean = Conv2DTranspose(3, (6, 6), strides= (2,2), activation='sigmoid')(deconv)

encoder = Model(img_input, z_mean, name='encoder')
decoder = Model(decoder_input, decoded_mean, name='decoder')
decoder.summary()
reconstructed = decoder(z)
vae = Model(img_input, reconstructed, name='vae')
opt = RMSprop(lr=0.00025)
vae.compile(optimizer=opt, loss=vae_loss)
vae.summary()

# we instantiate these layers separately so as to reuse them later

#decoder_h = Dense(intermediate_dim, activation='relu')
#decoder_mean = Dense(original_dim, activation='sigmoid')
#h_decoded = decoder_h(z)
#x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
#vae = Model(x, x_decoded_mean)

# Compute VAE loss
#xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#vae_loss = K.mean(xent_loss + kl_loss)

#vae.compile(loss= vae_loss, optimizer='rmsprop')
#vae.summary()


# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = y_train = x_test = y_test = getImages()

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
try:
    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))
finally:
    encoder.save('models/encoder_01.h5')
    decoder.save('models/decoder_01.h5')
    vae.save('models/vae_01.h5')

# build a model to project inputs on the latent space
#encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()

# build a digit generator that can sample from the learned distribution
#decoder_input = Input(shape=(latent_dim,))
#_h_decoded = decoder_h(decoder_input)
#_x_decoded_mean = decoder_mean(_h_decoded)
#generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 64
figure = np.zeros((digit_size * n, digit_size * n, 3))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        #z_sample = np.array([[xi, yi, xi, yi]])
        z_sample = np.random.normal(0, 1, (1, latent_dim))
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size, 3)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size, :] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()