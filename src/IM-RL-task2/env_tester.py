'''
This script loads an environment model implemented as an MDN and runs it in closed loop after seeding it with an image.
Actions are selected by a pre-trained controller that is also loaded along with an encoder to encode the seeding image
from the environment.
'''

import numpy as np
import random
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from load_process_images import getImages
from mdn import MDN
import cv2

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 1
LATENT_DIM = 8
NUM_COMPONENTS = 5

VAE_VER = '0008_1'
MODEL_VER = '0001'

def int2onehot(a, n):
    onehot = np.zeros(n)
    onehot[a] = 1
    return onehot

def onehot2int(onehot):
    return np.argmax(onehot)

def mse(x,y):
    return ((x-y)**2).mean()

class ArrowEnv:
    def __init__(self, num_items=3, use_all=True, val=False, stochastic_gestures=False, stochastic_dynamics=False):
        self.arrow_state = []
        self.gest_state = None
        self.num_items = num_items
        self.stoch_gest = stochastic_gestures
        self.stoch_dyn = stochastic_dynamics
        self.stoch_gest_prob = 0.1
        self.imgs_dict = dict(getImages(return_single=False, use_all=use_all, val=val))
        self.s_bar = None

        self.env_model = MDN(num_components=NUM_COMPONENTS, in_dim=LATENT_DIM + 6, out_dim=LATENT_DIM,
                             model_path="models/env_model_" + MODEL_VER + ".h5")
        self.encoder = load_model('models/encoder_' + VAE_VER + ".h5")
        self.dqn_model = load_model('models/controller_' + MODEL_VER + ".h5")
        self.decoder = load_model("models/decoder_" + VAE_VER + ".h5")
        self.r_model = load_model("models/r_model_" + MODEL_VER + ".h5")

    def reset(self):
        #self.arrow_state is the internal state.
        # numbers in the arrow_state list are the states of the arrows
        # arrow states key: 0 = U, 1 = L, 2 = D, 3 = R
        # gest_state is which arrow is being pointed to
        self.arrow_state = []
        self.arrow_state = random.sample(range(self.num_items), 3)  #samples with no replacement to guarantee unique config
        self.gest_state = random.choice(range(self.num_items))
        while self._isSolved(): # avoid having the initial state already solved on reset
            self.gest_state = random.choice(range(self.num_items))
        return self._generateState()


    def step(self, action):
        #actions key: each 2 successive values represent rotating an object counerCW or CW respectively.
        #exanple: 0 = rotate item 0 CCW, 1 = rotate item 1 CW, 2 = rotate item 1 CCW etc...

        assert action<self.getActSpaceSize() and action>=0, \
            "action cannot exceed num_items*2 or be less than 0, action = %r" % action

        done = 0

        if action == 0:
            self.arrow_state[0] = (self.arrow_state[0] + 1) % 4
        elif action == 1:
            self.arrow_state[0] = (self.arrow_state[0] - 1) % 4
        elif action == 2:
            self.arrow_state[1] = (self.arrow_state[1] + 1) % 4
        elif action == 3:
            self.arrow_state[1] = (self.arrow_state[1] - 1) % 4
        elif action == 4:
            self.arrow_state[2] = (self.arrow_state[2] + 1) % 4
        elif action == 5:
            self.arrow_state[2] = (self.arrow_state[2] - 1) % 4

        # now compute reward
        if len(self.arrow_state) > len(set(self.arrow_state)):  # non-unique config
            reward = -10
            done = 1
            print("Non-unique config!")
        elif self._isSolved():   # only arrow pointed to is up
            reward = 50
            done = 1
            print("Solved!")
        else:
            reward = -1

        # pointing has a random chance to change
        if self.stoch_gest and (random.random() <= self.stoch_gest_prob) and (done != 1):
            gest_aslist = []
            gest_aslist.append(self.gest_state)
            self.gest_state = random.choice(list(set(range(self.num_items)) - set(gest_aslist)))
            # set difference makes sure the new gesture is different

        return self._generateState(), reward, done

    def _generateState(self):
        # returns observable state
        state = self._getStateString()
        imgs = self.imgs_dict[state]
        return random.choice(imgs)

    def printState(self):
        print ("State: " + self._getStateString())

    def getStateSpaceSize(self):
        return (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)

    def getActSpaceSize(self):
        return self.num_items*2

    def _isSolved(self):
        return self.arrow_state[self.gest_state] == 0

    def _getStateString(self):
        state_str = ''
        for i in range(len(self.arrow_state)):
            if self.arrow_state[i] == 0:
                state_str += 'U'
            elif self.arrow_state[i] == 1:
                state_str += 'L'
            elif self.arrow_state[i] == 2:
                state_str += 'D'
            elif self.arrow_state[i] == 3:
                state_str += 'R'
        state_str += str(self.gest_state)
        return state_str
    def encode(self, s):
        encoded = np.asarray(self.encoder.predict(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))))
        return encoded[0, 0, :]

    def model_reset(self, s_zero):
        print('resetting model...')
        self.s_bar = self.encode(s_zero)
        return self.s_bar

    def model_step(self, a):
        #print self.s_bar.shape
        s_a = np.append(self.s_bar, int2onehot(a,self.getActSpaceSize()))
        #print s_a.shape
        mu, sigma, pi = self.env_model.get_dist_params(s_a.reshape(1, -1))
        #print('model out = ', model_out)
        print('coefficients = ', pi)
        component = np.random.choice(np.arange(0, NUM_COMPONENTS, 1), p=pi.flatten())
        mu = mu[0, component, :]
        sigma = sigma[0, component, :]
        #z = np.random.normal(mu, sigma)
        # z = np.zeros([LATENT_DIM,])
        # for i in range(NUM_COMPONENTS):
        #     z_log_var = z_log_vars[:, i * LATENT_DIM:(i + 1) * LATENT_DIM]
        #     component = np.random.normal(loc=z_means[:, i], scale=np.exp(z_log_vars[:, i]/2))
        #     z = z + component*coefficients[i]
        #self.s_bar = self.s_bar + z
        self.s_bar = mu
        #self.s_bar = model_out
        r_out = self.r_model.predict(s_a.reshape((1,s_a.size)))
        r = r_out[0].flatten()
        done = r_out[1].flatten()
        #print('means = ', mu)
        #print('sigmas = ', sigma)

        return self.s_bar, r, done

    def act(self, s):
        out = self.dqn_model.predict(np.reshape(s, (1,LATENT_DIM)))
        return np.argmax(out)


testEnv = ArrowEnv()
s = testEnv.reset()
s_hat = testEnv.model_reset(s)
d = d_hat = 0
episodes = 0
MAX_EPISODES = 50
log = []
misclass_r = 0
misclass_d = 0
im_size = 64
figure = np.zeros((im_size, im_size, 3))
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
while(episodes < MAX_EPISODES):
    if round(d_hat) == 2:
        #print 'new episode'
        episodes = episodes + 1
        d_hat = 0
        s = testEnv.reset()
        s_hat = testEnv.model_reset(s)
    figure[:, 0:im_size, :] = s
    decoded = testEnv.decoder.predict(s_hat.reshape(1, LATENT_DIM))
    figure[:, 0:im_size, :] = decoded
    #print 'mse(s): ', mse(testEnv.encode(s), s_hat)#, ', r: ', r, ', r_hat', r_hat, ', d: ', d, ', d_hat', d_hat
    #print('s_hat', s_hat)

    #a = testEnv.act(testEnv.encode(s))
    cv2.imshow('image', figure)
    cv2.waitKey(10)
    a = int(raw_input('Enter action:'))
    if a > 5:
        a = testEnv.act(s_hat)
    print ('z = ', s_hat, 'a = ', a)
    #s, r, d = testEnv.step(a)
    #s_hat, r_hat, d_hat = testEnv.model_step(a)
    s_hat, r_hat, d_hat = testEnv.model_step(a)
    #print ('action = ', a)

    #print 'r: ', r, ', r_hat', r_hat, ', d: ', d, ', d_hat', d_hat
    print 'r_hat: ', r_hat, 'd_hat: ', d_hat
    #print 'mse:  s: ', mse(testEnv.get_sbar(s),s_hat), ', r: ' , mse(r,r_hat) , ', d: ' , mse(d,d_hat)

    #if (round(r)-round(r_hat)) != 0:
        #misclass_r += 1
    #if (round(d)-round(d_hat)) != 0:
        #misclass_d += 1
    #log.append([mse(testEnv.encode(s), s_hat), mse(r, r_hat), mse(d, d_hat)])
#print 'misclass(r) = ', misclass_r, ' , misclass(d) = ', misclass_d







