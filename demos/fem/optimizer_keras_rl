import numpy as onp
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

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
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.

model = Sequential()
model.add(Convolution2D(16,(3,3),padding='same',activation='relu'))
model.add(Convolution2D(8,(3,3),padding='same',activation='relu'))
model.add(Convolution2D(4,(3,3),padding='same',activation='relu'))
model.add(Convolution2D(1,(3,3),padding='same',activation='relu'))
model.add(Flatten())
    
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=128, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=256,
               target_model_update=100, policy=policy)
dqn.compile(Adam(lr=2.5e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=36, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format("ADOPT"), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
