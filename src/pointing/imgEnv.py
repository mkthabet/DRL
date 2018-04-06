import numpy as np
import random
import cv2
import os

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 2

def processImage( img ):
    #rgb = None
    rgb = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return rgb
    #r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance
    #o = gray.astype('float32') / 128 - 1    # normalize
    #return o

class PointingEnv:
    def __init__(self, num_items = 2):
        self.num_items = num_items
        self.act_space_size = self.num_items
        self.b_imgs, self.g_imgs, self.b_only, self.g_only = [], [], [], []
        for filename in os.listdir('b'):
            img = cv2.imread(os.path.join('b',filename))
            if img is not None:
                self.b_imgs.append(processImage(img))
        for filename in os.listdir('g'):
            img = cv2.imread(os.path.join('g',filename))
            if img is not None:
                self.g_imgs.append(processImage(img))
        for filename in os.listdir('b_only'):
            img = cv2.imread(os.path.join('b_only', filename))
            if img is not None:
                self.b_only.append(processImage(img))
        for filename in os.listdir('g_only'):
            img = cv2.imread(os.path.join('g_only', filename))
            if img is not None:
                self.g_only.append(processImage(img))


    def reset(self):
        c = random.randint(0,1) #0 = b, 1 = g, 2 = b_only, 3 = g_only
        self.state = c
        return self._generateState()


    def step(self, action):
        #action {1, ..., n-1, n} where n=num_items is a position in the list starting from 1.
        #positions start from 0, item id's start from 1
        assert action<self.num_items and action>=0, "action cannot exceed number of items or be less than 0, action = %r" % action

        done = 0
        if self.state == action:
            reward = 1
        else:
            reward = 0

        return self._generateState(), reward, done

    def _generateState(self):
        #0 = b, 1 = g
        if self.state == 0:
            return random.choice(self.b_imgs)
        else:
            return random.choice(self.g_imgs)


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
        return self.num_items


#testEnv = PointingEnv()
#img = testEnv.reset()
#cv2.imshow('test', img)
#cv2.waitKey(0)
#img, r, d = testEnv.step(1)
#cv2.imshow('test', img)
#cv2.waitKey(0)


