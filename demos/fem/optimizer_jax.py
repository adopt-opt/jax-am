from typing import Type
import numpy as onp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import gym

import haiku as hk
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
env = TopOptEnv(size_x=Nx, size_y=Ny, render_mode="human", jax_model=simulator)
init(autoreset=True)
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
#         self.model = torch.nn.Sequential()

#         self.model.add(torch.nn.Conv2D(3, 16,(3,3),padding='same',activation='relu'))
#         self.model.add(torch.nn.Conv2D(16, 8,(3,3),padding='same',activation='relu'))
#         self.model.add(torch.nn.Conv2D(8, 4,(3,3),padding='same',activation='relu'))
#         self.model.add(torch.nn.Conv2D(4, 1,(3,3),padding='same',activation='relu'))
#         self.model.add(torch.nn.Flatten())
    
#     def forward(self, state):
#         # x = self.model(state)
    
#         # #V = self.model_V(x)
#         # #A = self.model_A(x)
        
#         # Q = x#V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))
#         return self.model(state)

class Double_DQN(hk.Module):
    def _init_(self):
        super()._init_(name="CNN")
        self.conv_model = hk.transform(hk.Sequential([
                                        hk.Conv2D(output_channels=16, kernel_shape=(3,3), padding="SAME"),
                                        jax.nn.relu,

                                        hk.Conv2D(output_channels=8, kernel_shape=(3,3), padding="SAME"),
                                        jax.nn.relu,

                                        hk.Conv2D(output_channels=4, kernel_shape=(3,3), padding="SAME"),
                                        jax.nn.relu,

                                        hk.Conv2D(output_channels=1, kernel_shape=(3,3), padding="SAME"),
                                        jax.nn.relu,

                                        hk.Flatten(),
                                    ])) 
    def __call__(self, x_batch):
        return self.conv_model(x_batch)
    
def DQN_eval(x):
    mod = Double_DQN()
    return mod(x)

def DQN_pred(x):
    mod = Double_DQN()
    return mod(x)

class Agent():
    def __init__(self,env, eps_dec, eps_min, batch_size, lr, mem_size, n_actions, input_dims, epsilon, EX, EY, gamma):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions=n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.EX=EX
        self.EY=EY
        self.env=env
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        #self.replace = opts.replace
        self.batch_size = batch_size
        self.lr=lr
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        
        # self.q_eval = hk.transform(Double_DQN()) 
        # self.q_next = hk.transform(Double_DQN()) 

        self.q_eval = hk.transform(DQN_eval) 
        self.q_next = hk.transform(DQN_pred)

        print('models initialized')
        
        # #this_dir, this_filename = os.path.split(__file__)
        # self.checkpoint_file_save=this_dir+'/NN_Weights/'+filename_save+'_NN_weights'
        # self.checkpoint_file_load=this_dir+'/NN_Weights/'+filename_load+'_NN_weights'
        # self.q_eval.compile(optimizer=Adam(learning_rate=self.lr),
        #                     loss='mean_squared_error')
        # # just a formality, won't optimize network
        # self.q_next.compile(optimizer=Adam(learning_rate=self.lr),
        #                     loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation,load_checkpoint,Testing):
        self.action_space = [i for i in range(self.n_actions)]
        if onp.random.random() < self.epsilon and not Testing:
            # Void=np.array(self.env.VoidCheck)
            # BC=np.array(np.reshape(self.env.BC_state,(1,(self.EX*self.EY))))
            # LC=np.array(np.reshape(self.env.LC_state,(1,(self.EX*self.EY))))
            # Clear_List=np.where(Void==0)[0]
            # BC_List=np.where(BC==1)[0]
            # LC_List=np.where(LC==1)[0]
            # self.action_space = [ele for ele in self.action_space if ele not in Clear_List]
            # self.action_space = [ele for ele in self.action_space if ele not in BC_List]
            # self.action_space = [ele for ele in self.action_space if ele not in LC_List]
            action =onp.random.choice(self.action_space)
        else:
            state = observation
            state=state.reshape(-1,self.EX,self.EY,3)
            actions = self.q_eval.call(state)
            action= onp.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            Loss=.5
            #return Loss

        if self.learn_step_counter % self.replace == 0 and self.learn_step_counter>0:
            self.q_next.set_weights(self.q_eval.get_weights())  

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_eval(states)
        self.q_pred=q_pred
        q_next = self.q_next(states_)
        q_target = q_pred.numpy()
        max_actions = onp.argmax(self.q_eval(states_), axis=1)
        # improve on my solution!
        for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
            q_pred[idx, actions[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        
        Loss= onp.subtract(q_target,q_pred.numpy())
        Loss= onp.square(Loss)
        Loss=Loss.mean()
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min 

        self.learn_step_counter += 1
        if self.learn_step_counter>5000:
            self.lr=2.5e-3
        if self.learn_step_counter>7500:
            self.lr=1e-3

        print(Loss)
        return Loss
        
    def save_models(self):
        print('... saving models ...')
        self.q_eval.save_weights(self.checkpoint_file_save)

    def load_models(self):
        print('... loading models ...')
        self.q_eval.load_weights(self.checkpoint_file_load).expect_partial()

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory =onp.zeros((self.mem_size, *input_shape),
                                        dtype= onp.float32)
        self.new_state_memory =onp.zeros((self.mem_size, *input_shape),
                                        dtype= onp.float32)
        self.action_memory =onp.zeros(self.mem_size, dtype= onp.int32)
        self.reward_memory =onp.zeros(self.mem_size, dtype= onp.float32)
        #self.terminal_memory =onp.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch =onp.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones


agent = Agent(env, eps_dec = 3.5e-4, eps_min = 1e-2, batch_size=128, lr=2.5e-3, mem_size=128, n_actions=36, input_dims=(3,6,6) ,epsilon = 1, EX = 6, EY = 6, gamma=0.99)

agent.learn()
