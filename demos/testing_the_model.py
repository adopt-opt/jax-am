from typing import Type
import numpy as onp
# import os


# import gym


import jax
import jax.numpy as jnp
from colorama import init, Fore, Back, Style

from problem import ProblemSetup
from env import TopOptEnv

import tensorflow as tf
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Layer
from keras.optimizers import Adam
import os

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class _build_model(tf.keras.Model):

    def __init__(self):
        super().__init__()
        #self.batch_size = batch_size
        self.conv = Sequential()
        self.conv.add(Conv2D(16,(3,3),padding='same' ,activation='relu', input_shape=state_size))
        self.conv.add(Conv2D(8,(3,3),padding='same',activation='relu'))
        self.conv.add(Conv2D(4,(3,3),padding='same',activation='relu'))
        self.conv.add(Conv2D(1,(3,3),padding='same', activation='relu'))

    def call(self, x):
        x = self.conv(x)
        x = x[:,::-1,:,:].reshape((x.shape[0], -1), order='F')
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, env:TopOptEnv, load_=False):
        self.state_size = state_size
        self.action_size = action_size
        #self.memory = deque(maxlen=6000)
        self.memory = deque(maxlen=30000)
        self.gamma = 0.1
        
        self.epsilon_decay = 3.5e-4
        self.epsilon_min = 0.01
        self.learning_rate = 2.5e-3
        self.e_start = 0
        self.epsilon = 0.9

        self.model = _build_model()
        self.model_target = _build_model()

        if load_:
            self.model.built = True
            # Specify which model to load here
            self.load(output_dir+'weights_2200.hdf5')
            #self.e_start = 2200
            #self.epsilon = 1 - self.e_start * self.epsilon_decay
            print(f'Model Succesfully Loaded')

        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        self.model_target.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

        self.registering_memory_step = 0
        self.env = env
    
    

    def remember(self, state, action, reward, next_state, done):
        self.registering_memory_step += 1
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size, episode_num):

        if len(self.memory) < batch_size:
            return 
            
        minibatch = random.sample(self.memory, batch_size)

        state_shape = (batch_size, self.state_size[0], self.state_size[1], self.state_size[2])
        states = np.zeros(state_shape)
        states_nxt = np.zeros(state_shape)
        actions, rewards, dones = [],[],[]

        for i in range(batch_size):
            states[i] = minibatch[i][0]
            states_nxt[i] = minibatch[i][3]
            actions.append(minibatch[i][1])
            rewards.append(minibatch[i][2])
            dones.append(minibatch[i][4])
        
        
        targets = onp.array(self.model(states)) 
        targets_nxt = self.model(states_nxt)
        
        targets_val = self.model_target(states_nxt)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                a_max = np.argmax(targets_nxt[i])
                targets[i][actions[i]] = rewards[i] + self.gamma * targets_val[i][a_max]
        
        self.model.fit(states, targets, epochs=1)
        
        self.new_eps = 1 - episode_num * self.epsilon_decay

        if self.new_eps > self.epsilon_min:
            self.epsilon = self.new_eps
        else:
            self.epsilon = self.epsilon_min
            

    def act(self, state, DQN_q_vals):
        # Filtering the action_space s.t. we only randomize from legal actions
        new_action_space = range(36)
        new_action_space = [ele for ele in new_action_space if ele not in self.env.bounded_cells]
        new_action_space = [ele for ele in new_action_space if ele not in self.env.loaded_cells]
        if onp.random.rand() <=0.4:
            new_action_space = [ele for ele in new_action_space if ele not in self.env.removed_cells]
        if onp.random.rand() <= self.epsilon:
            print(f"\n############## RANDOM ACTION ##############")
            return random.choice(new_action_space), DQN_q_vals
        print(f"\n############## DQN ACTION ##############")
        
        state_inp = onp.array(state[onp.newaxis, :, :, :])
        act_values = self.model(state_inp)

        new_act_values= onp.array(act_values).squeeze(0)
        print(f'New act Values : {new_act_values}')
        
        DQN_q_vals.append(onp.argmax(new_act_values))
        
        return onp.argmax(new_act_values), DQN_q_vals
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.built = True
        self.model.load_weights(name)


state_size = (6,6,3)
action_size = 36
batch_size = 128
output_dir = 'model_output4/'

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
num_steps = num_of_cells - (num_bounded_cell + num_loaded_cell)
# instance definition
simulator = ProblemSetup(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, num_bounded_cell=num_bounded_cell, num_loaded_cell=num_loaded_cell, 
                        filled_density=filled_density, void_density=void_density, dim=dim, vec=vec)

env = TopOptEnv(size_x=Nx, size_y=Ny, render_mode="human", jax_model=simulator)
agent = DQNAgent(state_size, action_size, env)

agent.load(output_dir + "weights_" + "4750" + ".hdf5")

n_episodes = 500
init(autoreset=True)
for e in range(n_episodes):
    state = env.reset()
    done = False
    time = 0
    while not done:
    # env.render()
        act_values = agent.model.predict(onp.array(state[onp.newaxis, :, :, :]))
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes-1, reward, agent.epsilon))
        time += 1
        new_act_values = onp.ravel(onp.array([new[::-1] for new in onp.reshape(act_values, (env.size_x, env.size_y)).T]))
        # Filtering the action_values s.t. we firstly assign 0 to every impossible action then take the legal action with the highest score
        for action_index in range(new_act_values.shape[0]):
            if action_index in agent.env.bounded_cells or action_index in agent.env.loaded_cells or action_index in agent.env.removed_cells:
                new_act_values[action_index] = 0

        action = onp.argmax(new_act_values)

        state, action, reward, next_state, done = env.step(action)

