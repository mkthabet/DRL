

import random, numpy, math, gym

#-------------------- BRAIN ---------------------------
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from pointing_env import PointingEnv

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3
LATENT_DIM = 4
sortedCnt = 0

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.controller, self.encoder = self._createModel()

    def _createModel(self):
        controller = load_model('models/controller_400.h5')
        encoder = load_model('models/encoder_105.h5')

        return controller, encoder

    def predict(self, s):
            return self.controller.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, LATENT_DIM)).flatten()

    def encode(self, s):
        encoded = np.asarray(self.encoder.predict(s))
        return encoded[0, 0, :]

BATCH_SIZE = 64


class Agent:
    steps = 0

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        
    def act(self, s):
        return numpy.argmax(self.brain.predictOne(s))


#-------------------- ENVIRONMENT ---------------------
done_cnt = 0
failed_cnt = 0
class Environment:
    def __init__(self, num_items):
        self.env = PointingEnv(num_items)

    def run(self, agent, inspect = False):
        s = self.env.reset()
        R = 0
        global failed_cnt
        global done_cnt
        while True:
            sbar = agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
            a = agent.act(sbar)
            print a
            s_, r, done = self.env.step(a)

            s = s_
            R += r

            if done:
                if r == -1:
                    failed_cnt += 1
                else:
                    done_cnt += 1
                break

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
num_items = 3
env = Environment(num_items)

stateCnt  = env.env.getStateSpaceSize()
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)

episodes = 0
MAX_EPISODES = 50
while episodes < MAX_EPISODES:
    env.run(agent)
    episodes += 1
    #agent.brain.model.save("point_3.h5")
#env.run(agent, False)
print('Done count: ', done_cnt)
print('failed count: ', failed_cnt)
