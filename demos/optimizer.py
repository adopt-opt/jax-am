from typing import Type
import numpy as onp

import jax
import jax.numpy as jnp

from colorama import init, Fore, Back, Style

from problem import ProblemSetup
#from env import TopOptEnv

from env_withgif import TopOptEnv
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

#from tensorflow.python.platform import tf_logging as logging
import logging

#logging._get_logger().propagate = False
mylog = logging.getLogger()
mylog.addHandler(logging.StreamHandler())
mylog.setLevel(logging.INFO)
mylog.addHandler(logging.FileHandler('./run.log'))
#mylog.propagate = False
# log and log
def log(s):
    #log(s)
    mylog.info(s)






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

simulator = ProblemSetup(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, num_bounded_cell=num_bounded_cell, num_loaded_cell=num_loaded_cell, 
                        filled_density=filled_density, void_density=void_density, dim=dim, vec=vec)


env = TopOptEnv(size_x=Nx, size_y=Ny, render_mode="human", jax_model=simulator)
init(autoreset=True)

state_size = (6,6,4)
action_size = 36
batch_size = 128
#batch_size = 256
#n_episodes = 1000
output_dir = 'model_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

avg_q_values = []
class StateTransformationLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        inputs = tf.array(inputs[-1, :, :, -1])
        return tf.reshape(tf.rot90(inputs, k=1, axes=(1, 0)), (-1))

class _build_model(tf.keras.Model):

    def __init__(self, output_filtering=False):
        super().__init__()
        self.conv = Sequential()
        self.output_filtering = output_filtering
        self.conv.add(Conv2D(16,(3,3),padding='same' ,activation='relu', input_shape=state_size))
        self.conv.add(Conv2D(8,(3,3),padding='same',activation='relu'))
        self.conv.add(Conv2D(4,(3,3),padding='same',activation='relu'))
        self.conv.add(Conv2D(1,(3,3),padding='same',activation='relu'))

    def call(self, x):
        #print(f' x shape: {x[:,:,:,-1][:,:,:,np.newaxis].shape}')
        #print(f' self conv x: {self.conv(x).shape}')
        #exit(1)
        x1 = self.conv(x) 
        if self.output_filtering:
            x1*= x[:,:,:,-1][:,:,:,np.newaxis]
        x1 = x1[:,::-1,:,:].reshape((x.shape[0], -1), order='F')
        return x1
    
class _build_model(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv = Sequential()
        self.conv.add(Conv2D(16,(3,3),padding='same' ,activation='relu', input_shape=state_size))
        self.conv.add(Conv2D(8,(3,3),padding='same',activation='relu'))
        self.conv.add(Conv2D(4,(3,3),padding='same',activation='relu'))
        self.conv.add(Conv2D(1,(3,3),padding='same',activation='relu'))

    def call(self, x):
        x1 = self.conv(x) 
        return x1

class DQNAgent:
    def __init__(self, state_size, action_size, env:TopOptEnv, load_=False, output_filtering=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=30000)
        self.gamma = 0.1
        
        self.epsilon_decay = 3.5e-4
        self.epsilon_min = 0.01
        self.learning_rate = 2.5e-3
        self.e_start = 0
        self.epsilon = 0.9

        self.model = _build_model(output_filtering=output_filtering)
        self.model_target = _build_model(output_filtering=output_filtering)

        if load_:
            self.model.built = True
            # Specify which model to load here
            self.load(output_dir+'weights_9800.hdf5')
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
        
        self.model.fit(states, targets, epochs=2)
        
        self.new_eps = 1 - episode_num * self.epsilon_decay

        if self.new_eps > self.epsilon_min:
            self.epsilon = self.new_eps
        else:
            self.epsilon = self.epsilon_min
            

    def act(self, state, DQN_q_vals):
        # Filtering the action_space s.t. we only randomize from legal actions
        
        ## Testing
        
        #state_vm_vals = state[::-1,:,0].reshape(-1, order='F')
        #print(f'state vm vals: {state_vm_vals}')
        #return onp.argmax(state_vm_vals), DQN_q_vals
        
        new_action_space = range(36)
        new_action_space = [ele for ele in new_action_space if ele not in self.env.bounded_cells]
        new_action_space = [ele for ele in new_action_space if ele not in self.env.loaded_cells]
        if onp.random.rand() <=0.9:
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
        self.model.load_weights(name)

load_ = False
o_f = True
agent = DQNAgent(state_size, action_size, env, load_=load_, output_filtering=o_f)

n_episodes = 10000
DQN_avg_q_vals = []

# for e in range(agent.e_start, n_episodes):
#     state = env.reset()
#     done = False
#     time = 0
#     DQN_q_vals = []
#     while not done:
#         action, DQN_q_vals = agent.act(state, DQN_q_vals)
#         state, action, reward, next_state, done = env.step(action, e)

#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         print(f'REWARD : {reward}')
#         if done:
#             print("episode: {}/{}, score: {}, eps: {:.2}".format(e, n_episodes-1, reward, agent.epsilon))
#             DQN_avg_q_vals.append(onp.sum(onp.array(DQN_q_vals))/len(DQN_q_vals))
#         time += 1
#     # Save q vals in csv for easy reading and plotting
#     data = onp.asarray(DQN_avg_q_vals)
#     onp.savetxt('Avg_q_vals_per_episode.csv', data, delimiter=',')

DQN_avg_q_vals = []
reward_avg_q_vals = []
print(f'Gamma : {agent.gamma}')
print(f'Output filtering: {o_f}')
for e in range(agent.e_start, n_episodes):
    state = env.reset()
    done = False
    time = 0
    DQN_q_vals = []
    reward_vals=[]

    while not done:
        action, DQN_q_vals = agent.act(state, DQN_q_vals)
        state, action, reward, next_state, done = env.step(action, e)
        reward_vals.append(reward)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        print(f'REWARD : {reward}')
        if done:
            print("episode: {}/{}, score: {}, eps: {:.2}".format(e, n_episodes-1, reward, agent.epsilon))
            DQN_avg_q_vals.append(onp.sum(onp.array(DQN_q_vals))/len(DQN_q_vals))
            reward_avg_q_vals.append(onp.sum(onp.array(reward_vals))/len(reward_vals))

        time += 1
    data = onp.asarray(reward_avg_q_vals)
    onp.savetxt('Average_reward_per_episode.csv', data, delimiter=',')

    agent.train(batch_size, e)

    if e % 100 ==0 and e>1:
        agent.model_target.set_weights(agent.model.get_weights())

    if e % 50 == 0 or e == n_episodes-1:
        print(f"\nLen of memory = {len(agent.memory)}")
        if load_:
            output_dir = 'model_output_loaded/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

