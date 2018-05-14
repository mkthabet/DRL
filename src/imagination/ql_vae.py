import random, numpy, math, gym

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout
from keras.optimizers import *
from keras.models import Model, model_from_json, load_model
from imgEnv import *

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 3

LATENT_DIM = 3

ENV_LEARN_START = 0   #number of episodes before train_controllering env model starts`

def mse(x,y):
    return ((x-y)**2).mean()

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.controller, self.env_model, self.encoder, self.controller_target = self._createModel()

    def _createModel(self):
        encoder = load_model('models/encoder_14.h5')

        controller_input = Input(shape=(LATENT_DIM,), name='controller_input')
        controller_out = Dense(units=512, activation='relu')(controller_input)
        controller_out = Dense(units=actionCnt, activation='linear')(controller_out)
        controller = Model(inputs=controller_input, outputs=controller_out)
        opt = RMSprop(lr=0.00025)
        controller.compile(loss='mse', optimizer=opt)
        #controller.summary()

        #just copy the architecure
        json_string = controller.to_json()
        controller_target = model_from_json(json_string)

        env_model_input = Input(shape=(LATENT_DIM+1,), name = 'env_in')
        #print 'env in shape', env_in_shape
        env_out = Dense(units=512, activation='relu', name = 'env_dense1')(env_model_input)
        #env_dropout1 = Dropout(0.5)
        #env_out = env_dropout1(env_out)
        env_out = Dense(units=256, activation='relu', name = 'env_dense2')(env_out)
        #env_dropout2 = Dropout(0.5)
        #env_out = env_dropout2(env_out)
        env_out = Dense(units=LATENT_DIM+2, activation='linear', name = 'env_out')(env_out)
        env_model = Model(inputs=env_model_input, outputs=env_out)
        opt_env = RMSprop(lr=0.00025)
        env_model.compile(loss='mse', optimizer=opt)

        return controller, env_model, encoder, controller_target

    def train_controller(self, x, y, epoch=1, verbose=0):

        self.controller.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def train_env(self, x, y, epoch=1, verbose=0):
        self.env_model.fit(x, y, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.controller_target.predict(s)
        else:
            return self.controller.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, LATENT_DIM), target).flatten()

    def encode(self, s):
        return self.encoder.predict(s)

    def updateTargetModel(self):
        self.controller_target.set_weights(self.controller.get_weights())
        

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ , d)
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

MAX_EPSILON = 0.6
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1

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

    def observe(self, sample):  # in (s, a, r, s_, done) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

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
        a_vec = numpy.array([ o[1] for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=False)
        #s_bar = agent.brain.encode(states)
        #print 'sbar' , s_bar.shape
        #s_bar_= agent.brain.encode(states_)

        x = numpy.zeros((len(batch), LATENT_DIM))
        y = numpy.zeros((len(batch), self.actionCnt))

        x_env = numpy.zeros((len(batch), LATENT_DIM+1))
        #print 'xenv' , x_env.shape
        y_env = numpy.zeros((len(batch), LATENT_DIM+2))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]; done = o[4]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t
            #print 'sbar[i]', s_bar[i].shape
            x_env[i] = np.append(states[i], a)
            y_env[i] = np.append(states_[i], [r, done])

        self.brain.train_controller(x, y)

        if episodes>ENV_LEARN_START:
            self.brain.train_env(x_env, y_env)


#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, num_items):
        self.env = PointingEnv(num_items)

    def run(self, agent, inspect = False):
        s = self.env.reset()
        R = 0 
        global sortedCnt
        while True:         
            if inspect: self.env.printState()

            sbar = np.asarray(agent.brain.encode(np.reshape(s, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))))
            sbar = sbar[0, :, :].flatten()    #get means of latent space distribution

            a = agent.act(sbar)
            s_, r, done = self.env.step(a)

            sbar_ = np.asarray(agent.brain.encode(np.reshape(s_, (1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))))
            sbar_ = sbar_[0, :, :].flatten()    #get means of latent space distribution
            if done: # terminal state
                s_ = None

            agent.observe( (sbar, a, r, sbar_,done) )
            agent.replay()

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R, ", episode: ", episodes)

#-------------------- MAIN ----------------------------
num_items = 2;
env = Environment(num_items)

stateCnt  = env.env.getStateSpaceSize()
actionCnt = env.env.getActSpaceSize()

agent = Agent(stateCnt, actionCnt)

episodes = 0
MAX_EPISODES = 1000

try:
    while episodes < MAX_EPISODES:
        env.run(agent)
        episodes = episodes + 1
finally:
    ss=0
    agent.brain.controller.save("models/controller_100.h5")
    agent.brain.env_model.save("models/env_model_100.h5")
#env.run(agent, False)
