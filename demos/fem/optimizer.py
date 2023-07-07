from typing import Type
import numpy as onp
# import os


# import gym


import jax
import jax.numpy as jnp

# import tensorflow  as tf
# from tensorflow import keras

#import torch

# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam

#from keras.utils import generic_utils

# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory

from colorama import init, Fore, Back, Style

from problem import ProblemSetup
from env import TopOptEnv

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import os

# constant decleration for problem setup
Nx, Ny = 6, 6
Lx, Ly = 6, 6
num_bounded_cell = 2
num_loaded_cell = 1
filled_density = 1.
void_density = 1e-4
dim = 2
vec = 2
# design variable initialization
num_of_cells = Nx * Ny
vf = 1
init_rho_vector = vf*onp.ones((num_of_cells, 1)) 
# optimization paramaters decleration
num_episodes = 10
num_steps = num_of_cells - (num_bounded_cell + num_loaded_cell)
# instance definition
simulator = ProblemSetup(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, num_bounded_cell=num_bounded_cell, num_loaded_cell=num_loaded_cell, 
                        filled_density=filled_density, void_density=void_density, dim=dim, vec=vec)


# Get the environment and extract the number of actions.

#onp.random.seed(123)
#env.seed(123)
#nb_actions = env.action_space.n

# Next, we build a very simple model.

# model = models.Sequential()
# model.add(layers.Conv2D(16,(3,3),padding='same',activation='relu'))
# model.add(layers.Conv2D(8,(3,3),padding='same',activation='relu'))
# model.add(layers.Conv2D(4,(3,3),padding='same',activation='relu'))
# model.add(layers.Conv2D(1,(3,3),padding='same',activation='relu'))
# model.add(layers.Flatten())
    
# print(model.summary())

# # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# # even the metrics!
# memory = SequentialMemory(limit=128, window_length=1)
# policy = EpsGreedyQPolicy()
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=256,
#                target_model_update=100, policy=policy)
# dqn.compile(Adam(lr=2.5e-3), metrics=['mae'])

# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
# dqn.fit(env, nb_steps=36, visualize=False, verbose=2)

# # After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format("ADOPT"), overwrite=True)

# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)


## Implemetation nkbrown repo

# class Double_DQN(keras.Model):
#     def __init__(self):
#         super(Double_DQN, self).__init__()
#         self.model = keras.models.Sequential()
#         #self.model = torch.nn.Sequential()

#         self.model.add(keras.models.Conv2D(3, 16,(3,3),padding='same',activation='relu'))
#         self.model.add(keras.models.Conv2D(16, 8,(3,3),padding='same',activation='relu'))
#         self.model.add(keras.models.Conv2D(8, 4,(3,3),padding='same',activation='relu'))
#         self.model.add(keras.models.Conv2D(4, 1,(3,3),padding='same',activation='relu'))
#         self.model.add(torch.nn.Flatten())
    
#     def forward(self, state):
#         # x = self.model(state)
    
#         # #V = self.model_V(x)
#         # #A = self.model_A(x)
        
#         # Q = x#V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))
#         return self.model(state)

env = TopOptEnv(size_x=Nx, size_y=Ny, render_mode="human", jax_model=simulator)
init(autoreset=True)

state_size = 3*36
action_size = 36
batch_size = 8
n_episodes = 1000
output_dir = 'model_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=128)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 2.5e-3
        self.model = self._build_model()
    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(3, 16,(3,3),padding='same',activation='relu'))
        model.add(Conv2D(16, 8,(3,3),padding='same',activation='relu'))
        model.add(Conv2D(8, 4,(3,3),padding='same',activation='relu'))
        model.add(Conv2D(4, 1,(3,3),padding='same',activation='relu'))
        model.add(Flatten())
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward # if done
        
            if not done:
                target = (reward +  self.gamma * np.amax(self.model.predict(next_state)[0]))

            print("TEST STOP")
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)


agent = DQNAgent(state_size, action_size)

n_episodes = 1000
for e in range(n_episodes):
    state = env.reset()
    # state = np.reshape(state, [1, state_size])
    done = False
    time = 0
    while not done:
    # env.render()
        action = agent.act(state)
        state, action, reward, next_state, done = env.step(action)
        reward = reward if not done else -10
        # next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes-1, time, agent.epsilon))
        time += 1
    
    if len(agent.memory) > batch_size:
        print(f"Learning Started")
        agent.train(batch_size)
        print(f"Learning Stopped")

    
    if e % 20 == 0 and e > 1:
        print(len(agent.memory))
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")


