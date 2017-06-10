import numpy as np
from matplotlib import pyplot
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]


modelL = Sequential()
modelL.add(LSTM(nb_actions, input_shape=(1,) + env.observation_space.shape))
modelL.add(Activation('softmax'))
memoryL = EpisodeParameterMemory(limit=1000, window_length=1)
cemL = CEMAgent(model=modelL, nb_actions=nb_actions, memory=memoryL,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cemL.compile()
histL = cemL.fit(env, nb_steps=50000, visualize=False, verbose=1)
cemL.save_weights('cemL_{}_params.h5f'.format(ENV_NAME), overwrite=True)
cemL.test(env, nb_episodes=5, visualize=True)


modelD = Sequential()
modelD.add(Flatten(input_shape=(1,) + env.observation_space.shape))
modelD.add(Dense(nb_actions))
modelD.add(Activation('softmax'))
memoryD = EpisodeParameterMemory(limit=1000, window_length=1)
cemD = CEMAgent(model=modelD, nb_actions=nb_actions, memory=memoryD,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cemD.compile()
histD = cemD.fit(env, nb_steps=50000, visualize=False, verbose=1)
cemD.save_weights('cemD_{}_params.h5f'.format(ENV_NAME), overwrite=True)
cemD.test(env, nb_episodes=5, visualize=True)


pyplot.plot(histL.history['nb_steps'], histL.history['episode_reward'], linewidth=3, label='LSTM')
pyplot.plot(histD.history['nb_steps'], histD.history['episode_reward'], linewidth=3, label='Dense')
pyplot.grid()
pyplot.legend()
pyplot.xlabel('steps')
pyplot.ylabel('reward')
#pyplot.yscale('log')
pyplot.show()
