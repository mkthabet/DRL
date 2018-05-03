import numpy as np
import random
import cv2
import os
from keras.models import Model, load_model
from math import ceil, floor, fabs

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

def mse(x,y):
    return ((x-y)**2).mean()

def processImage( img ):
    rgb = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return rgb

def getDummyImg():
    filename = 'dummy_img.png'
    img = cv2.imread(os.path.join('dummy', filename))
    return processImage(img)

class PointingEnv:
    def __init__(self, num_items = 2):
        self.num_items = num_items
        self.act_space_size = self.num_items
        self.b_imgs, self.g_imgs, self.b_only, self.g_only, self.b_hand, self.g_hand = [], [], [], [], [], []
        VAL = 0
        val = 'validation/'
        if not VAL:
            val = ''   #change to empty string to test on training set
        for filename in os.listdir(val+'b'):
            img = cv2.imread(os.path.join(val+'b',filename))
            if img is not None:
                self.b_imgs.append(processImage(img))
        for filename in os.listdir(val+'g'):
            img = cv2.imread(os.path.join(val+'g',filename))
            if img is not None:
                self.g_imgs.append(processImage(img))
        for filename in os.listdir(val+'b_only'):
            img = cv2.imread(os.path.join(val+'b_only', filename))
            if img is not None:
                self.b_only.append(processImage(img))
        for filename in os.listdir(val+'g_only'):
            img = cv2.imread(os.path.join(val+'g_only', filename))
            if img is not None:
                self.g_only.append(processImage(img))
        for filename in os.listdir(val+'b_hand'):
            img = cv2.imread(os.path.join(val+'b_hand', filename))
            if img is not None:
                self.b_hand.append(processImage(img))
        for filename in os.listdir(val+'g_hand'):
            img = cv2.imread(os.path.join(val+'g_hand', filename))
            if img is not None:
                self.g_hand.append(processImage(img))

        self.env_model = load_model("models/env_model_20.h5")
        self.conv_model = load_model("models/conv_model_20.h5")
        self.dqn_model = load_model('models/model_20.h5')

        self.s_bar = None


    def reset(self):
        #self.state is the internal state.
        #self.state = random.randint(0,1) #0 = b, 1 = g, 2 = b_only, 3 = g_only,
        self.state = 0
        return self._generateState()


    def step(self, action):
        assert action<self.getActSpaceSize() and action>=0, "action cannot exceed number of items +1 or be less than 0, action = %r" % action

        done = 0

        if self.state == 0:
            if action == 0:
                #self.state = random.choice([3, 5])
                self.state = 3
                reward = 1
            else:
                reward = -1
                done = 1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 1:
            if action == 1:
                #self.state = random.choice([2, 4])
                self.state = 4
                reward = 1
            else:
                reward = -1
                done = 1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 2:
            if action == 1:
                self.state = 0
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 3:
            if action == 0:
                self.state = 1
                reward = 0
            else:
                done = 1
                reward = -1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 4:
            if action == 2:
                done = 1
                reward = 5
            else:
                done = 1
                reward = -1
                print 'mistake.....'
            return self._generateState(), reward, done
        if self.state == 5:
            if action == 2:
                done = 1
                reward = 5
            else:
                done = 1
                reward = -1
                print 'mistake.....'

        return self._generateState(), reward, done

    def _generateState(self):
        # 0 = b, 1 = g, 2 = b_only, 3 = g_only, 4 = b_hand, 5 = g_hand
        #print 'state = ' , self.state
        if self.state == 0:
            return random.choice(self.b_imgs)
        elif self.state == 1:
            return random.choice(self.g_imgs)
        elif self.state == 2:
            return random.choice(self.b_only)
        elif self.state == 3:
            return random.choice(self.g_only)
        elif self.state == 4:
            return random.choice(self.b_hand)
        elif self.state == 5:
            return random.choice(self.g_hand)


    def printState(self):
        state = self._generateState()
        stateArr = np.array(state)
        print "state = %s" % stateArr.T
        #x = stateArr.shape
        #print "state dims: "
        #print(x)
        #for i in self.item_list: print i
        #print self.item_list
        #print state
        #for i in state: print i

    def getStateSpaceSize(self):
        return ( IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    def getActSpaceSize(self):
        return self.num_items+1

    def get_sbar(self, s):
        return self.conv_model.predict(np.expand_dims(s, axis=0))

    def model_reset(self, s_zero):
        self.s_bar = self.get_sbar(s_zero).flatten()

    def model_step(self, a):
        #print self.s_bar.shape
        s_a = np.append(self.s_bar, a)
        #print s_a.shape
        model_out = self.env_model.predict(s_a.reshape((1,1,s_a.size)))
        model_out = model_out.flatten()
        self.s_bar = model_out[:-2]
        r = model_out[-2]
        done = model_out[-1]

        return self.s_bar, r, done

    def act(self, s):
        out = self.dqn_model.predict(s.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)).flatten()
        return np.argmax(out)


testEnv = PointingEnv()
s = testEnv.reset()
#cv2.imshow('test', s)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#testEnv.model_reset(getDummyImg())0
testEnv.model_reset(s)
#ip = 0
d = 0
episodes = 0
MAX_EPISODES = 50
log = []
misclass_r = 0
misclass_d = 0
while(episodes < MAX_EPISODES):
    if d == 1:
        #print 'new episode'
        episodes = episodes + 1
        d = 0
        s = testEnv.reset()
        testEnv.model_reset(s)
    a = testEnv.act(s)
    s, r, d = testEnv.step(a)
    s_hat, r_hat, d_hat = testEnv.model_step(a)
    #print 'mse:  s: ', mse(testEnv.get_sbar(s),s_hat), ', r: ' , mse(r,r_hat) , ', d: ' , mse(d,d_hat)
    print 'mse(s): ', mse(testEnv.get_sbar(s), s_hat), ', r: ', r, ', r_hat', r_hat, ', d: ', d, ', d_hat', d_hat
    if (round(r)-round(r_hat)) != 0:
        misclass_r += 1
    if (round(d)-round(d_hat)) != 0:
        misclass_d += 1
    log.append([mse(testEnv.get_sbar(s), s_hat), mse(r, r_hat), mse(d, d_hat)])
print 'misclass(r) = ', misclass_r, ' , misclass(d) = ', misclass_d






