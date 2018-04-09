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
from toh_env import *

sortedCnt = 0

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model, self.rmodel = self._createModel()
        #self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        #model = Sequential()
        #model.add(Dense(output_dim=512, activation='relu', input_dim=stateCnt))
        #model.add(Dense(output_dim=512, activation='relu'))
        #model.add(Dense(output_dim=actionCnt, activation='linear'))
        model = load_model("toh_3-3_4.h5")
        rmodel = Sequential()
        rmodel.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
        rmodel.add(Dense(output_dim=64, activation='relu'))
        rmodel.add(Dense(output_dim=1, activation='linear'))
        opt = RMSprop(lr=0.00025)
        rmodel.compile(loss='mse', optimizer=opt)

        return model, rmodel

    def train(self, x, y, epoch=1, verbose=0):

        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def rtrain(self, x, y, epoch=1000, verbose=0):

        self.rmodel.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        # print "shape"
        # print(s.shape)
        #print "PREDICT:"
        #print self.model.predict(s)
        return self.model.predict(s)

    def predictOne(self, s):
        #print("state:", s)
      #  print " predictone:"
        #print self.predict(s.reshape(1, self.stateCnt)).flatten()
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

    def rpredict(self, s):
        return self.rmodel.predict(s)

    def rpredictOne(self, s):
        #print("state:", s)
      #  print " predictone:"
        #print self.predict(s.reshape(1, self.stateCnt)).flatten()
        return self.rpredict(s.reshape(1, self.stateCnt)).flatten()
        

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
BATCH_SIZE = 10

GAMMA = 0.99

MAX_EPSILON = 0.1
MIN_EPSILON = 0.05
LAMBDA = 0.0005      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        #self.epsilon = 0.1

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        xr = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        ry = numpy.zeros((batchLen, 1))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]

            ry[i] = r
            xr[i] = s_
            if r == -1:
                ry[i] = 0
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)
        self.brain.rtrain(xr, ry)
        #print "x", xr
        #print "ry", ry

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, num_items, num_stacks):
        self.env = Toh(num_items, num_stacks)

    def run(self, agent, inspect = False):
        s = self.env.reset()
        R = 0 
        global sortedCnt
        while True:         
            if inspect: self.env.printState()   
            a = agent.act(s)
            #print "action", a

            s_, r, done = self.env.step(a)
            

           # if done: # terminal state
             #   s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                if inspect: self.env.printState()
                if R>0:
                    sortedCnt += 1
                break

            if R<-500:
                #print "Min reward reached. Ending episode"
                break

            if R<-15:
                #print "Min reward reached. Ending episode"
                sortedCnt = 0

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
num_items = 3
num_stacks = 3
env = Environment(num_items,num_stacks)

stateCnt  = env.env.getStateSpaceSize()
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)
#i = 0
#try:
#    while i<100:
#        env.run(agent, inspect=False)
#        i+=1
#finally:
#    agent.brain.rmodel.save("rlearner.h5")
#env.run(agent, inspect=True)

#state = np.array([[0 ,0, 0],[3, 2, 1],[0, 0, 0]])
#state = np.array([[1, 0, 0],[2, 3, 0],[0, 0, 0]])
#print agent.brain.rpredictOne(state)

states = np.zeros((10,9))
states[0,:] = np.array([[ 2, 3, 0],[1, 0, 0],[0, 0, 0]]).flatten()    #r=0
states[1,:] = np.array([[ 3, 0, 0],[1, 0, 0],[2, 0, 0]]).flatten()    #r=0
states[2,:] = np.array([[ 3, 0, 0],[2, 1, 0],[0, 0, 0]]).flatten()     #r=-600
states[3,:] = np.array([[ 3, 0, 0],[1, 0, 0],[2, 0, 0]]).flatten()    #r=0
states[4,:] = np.array([[ 0, 0, 0],[3, 1, 0],[2, 0, 0]]).flatten()    #r=-600
states[5,:] = np.array([[ 0, 0, 0],[1, 0, 0],[3, 2, 0]]).flatten()    #r=-600
states[6,:] = np.array([[ 3, 0, 0],[0, 0, 0],[1, 2, 0]]).flatten()    #r=0
states[7,:] = np.array([[ 0, 0, 0],[0, 0, 0],[1, 2, 3]]).flatten()    #r=500
states[8,:] = np.array([[ 0, 0, 0],[1, 0, 0],[2, 3, 0]]).flatten()    #r = 0
states[9,:] = np.array([[ 2, 0, 0],[1, 0, 0],[3, 0, 0]]).flatten()    #r=0
#r = np.zeros((10,9))
#r[0,:] = [1]
r = np.array([[0],[0],[-600],[0],[-600],[-600],[0],[500],[0],[0]])

newstate1 = np.array([[ 0, 0, 0],[0, 0, 0],[3, 1, 2]]).flatten()
newstate2 = np.array([[ 0, 0, 0],[3, 0, 0],[ 1, 2, 0]]).flatten()
agent.brain.rtrain(states,r)

print agent.brain.rpredictOne(newstate2)