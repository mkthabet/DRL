# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use of a basic Q-network (without target network)
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# 
# author: Jaromir Janisch, 2016


#--- enable this to run on GPU
# import os    
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"  

import random, numpy, math, gym

#-------------------- BRAIN ---------------------------
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from itemQueue import *


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        #self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = load_model("sort_7.h5")
        return model

    def train(self, x, y, epoch=1, verbose=0):

        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        # print "shape"
        # print(s.shape)
        #print self.model.predict(s)
        return self.model.predict(s)

    def predictOne(self, s):
        #print("state:", s)
        #print self.predict(s.reshape(1, self.stateCnt)).flatten()
        return self.predict(s.reshape(1, self.stateCnt)).flatten()
        

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        
    def act(self, s):
        return numpy.argmax(self.brain.predictOne(s))

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, num_items):
        self.env = ItemQueue(num_items)

    def run(self, agent, inspect = False, init_list = []):
        s = self.env.reset(init_list)
        R = 0 
        while True:         
            if inspect: self.env.printState()   
            a = agent.act(s)

            s_, r, done = self.env.step(a)
            

            if done: # terminal state
                s_ = None   
                if inspect: self.env.printState()
                break

            s = s_
            R += r

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
num_items = 7;
env = Environment(num_items )

stateCnt  = env.env.getStateSpaceSize()
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)

init_list = list([ 7, 4, 2, 5, 3, 6, 1])

env.run(agent, inspect = True, init_list = init_list)

