import random, math, gym
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, Lambda, Concatenate
from keras.optimizers import *
from keras.models import Model, model_from_json, load_model
from keras import backend as K
from keras import metrics
# from pointing_model import PointingEnv
import matplotlib.pyplot as plt

IN_DIM = 1
OUT_DIM = 1
NUM_COMPONENTS = 24
BATCH_SIZE = None


class MDN:
    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM, num_components=NUM_COMPONENTS):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_components = num_components
        self.model, self.model_train = self._createModel()

    def _createModel(self):
        def get_mixture_coef(output, numComponents=24, outputDim=1):
            out_mu = output[:, 0:outputDim * numComponents]
            out_sigma = output[:, outputDim * numComponents:outputDim * numComponents * 2]
            out_pi = output[:, outputDim * numComponents * 2:]
            out_mu = K.reshape(out_mu, [-1, numComponents, outputDim])
            out_mu = K.permute_dimensions(out_mu, [1, 0, 2])  # shape = [numComponents, batch, outputDim]
            out_sigma = K.exp(out_sigma)
            return out_pi, out_sigma, out_mu

        def tf_normal(y, mu, sigma):
            oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)
            result = y - mu  # shape = [numComponents, batch, outputDim]
            result = K.permute_dimensions(result, [2, 1, 0])
            result = result * (1 / (sigma + 1e-8))
            result = -K.square(result) / 2
            result = K.exp(result) * (1 / (sigma + 1e-8)) * oneDivSqrtTwoPI
            result = K.prod(result, axis=[0])
            return result

        def get_lossfunc(out_pi, out_sigma, out_mu, y):
            result = tf_normal(y, out_mu, out_sigma)
            result = result * out_pi
            result = K.sum(result, axis=1, keepdims=True)
            result = -K.log(result + 1e-8)
            return K.mean(result)

        def mdn_loss(numComponents=24, outputDim=1):
            def loss(y, output):
                out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
                return get_lossfunc(out_pi, out_sigma, out_mu, y)
            return loss

        model_input = Input(shape=(1,), name='model_in')
        model_out = Dense(units=24, activation='tanh', name='model_dense4')(model_input)
        # model_out = Dense(units=2 * (1 * 24) + 24, name='model_out')(model_out)
        out_pi = Dense(units=NUM_COMPONENTS, activation='softmax', name='out_pi')(
            model_out)
        out_mu = Dense(units=OUT_DIM * NUM_COMPONENTS, name='out_mu')(model_out)
        out_sigma = Dense(units=OUT_DIM * NUM_COMPONENTS, name='out_sigma')(model_out)
        out_concat = Concatenate()([out_mu, out_sigma, out_pi])
        model_train = Model(inputs=model_input, outputs=out_concat)
        model = Model(inputs=model_input, outputs=[out_mu, out_sigma, out_pi])
        opt = Adam(lr=0.001)
        model_train.compile(loss=mdn_loss(), optimizer=opt)
        # model_train.summary()
        return model, model_train

    def train_model(self, x, y, epoch=2000, verbose=1):
        self.model_train.fit(x, y, batch_size=2500, epochs=epoch, verbose=verbose)

    def get_dist_params(self, x):
        model_out = np.asarray(self.model_train.predict(x))
        out_mu = model_out[:, 0:self.out_dim * self.num_components]
        out_sigma = model_out[:, self.out_dim * self.num_components:self.out_dim * self.num_components * 2]
        out_pi = model_out[:, self.out_dim * self.num_components * 2:]
        out_mu = np.reshape(out_mu, [-1, self.num_components, self.out_dim])
        out_sigma = np.reshape(out_sigma, [-1, self.num_components, self.out_dim])
        out_sigma = np.exp(out_sigma)
        #print 'out shape:', model_out.shape
        # means = model_out[0, :, :]
        #print 'means shape:', out_mu.shape
        # log_vars = model_out[1, :, :]
        #print 'vars shape:', out_sigma.shape
        # coefficients = model_out[2, :, :]
        #print 'coeffs shape:', out_pi.shape
        return out_mu, out_sigma, out_pi

def generate(out_mu, out_sigma, out_pi, testSize, numComponents=24, outputDim=1, M=1):
    out_mu = np.transpose(out_mu, [1, 0, 2])    #shape = [numComponents, batch, outputDim]
    result = np.random.rand(testSize, M, outputDim)
    rn = np.random.randn(testSize, M)
    mu = 0
    std = 0
    idx = 0
    for j in range(0, M):
        for i in range(0, testSize):
            for d in range(0, outputDim):
                idx = np.random.choice(24, 1, p=out_pi[i])
                mu = out_mu[idx, i, d]
                std = out_sigma[i, idx]
                result[i, j, d] = mu + rn[i, j] * std
    return result

def test():
    sampleSize = 250
    numComponents = 24
    outputDim = 1
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, sampleSize))).T
    r_data = np.float32(np.random.normal(size=(sampleSize, 1)))
    y_data = np.float32(np.sin(0.75 * x_data) * 7.0 + x_data * 0.5 + r_data * 1.0)
    # invert training data
    temp_data = x_data
    x_data = y_data
    y_data = temp_data

    x_test = np.float32(np.arange(-15.0, 15.0, 0.1))
    x_test = x_test.reshape(x_test.size, 1)
    mdn = MDN()
    mdn.train_model(x_data, y_data)
    mu, sigma, pi = mdn.get_dist_params(x_test)
    y_test = generate(mu, sigma, pi, x_test.size)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', x_test, y_test[:, :, 0], 'bo', alpha=0.3)
    plt.show()
####### MAIN #########

test()