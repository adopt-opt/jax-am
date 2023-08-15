import gym
import numpy as onp
import jax.numpy as np
import random
import pygame

from colorama import init, Fore, Back, Style
from problem import ProblemSetup

class TopOptEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size_x:int = 6, size_y:int = 6, render_mode=None, jax_model=None):
        # Dimensionality of the grid
        self.size_x, self.size_y = size_x, size_y 
        self.window_size = 512
        self.initial_rho_vector = onp.ones((self.size_x * self.size_y, 1))
        self.jax_model = jax_model
        self.points = self.jax_model.points
        self.cells = self.jax_model.cells
        
        #print(f'Jax model points : {self.jax_model.points}')
        #exit(1)

        # Our 3-dimensional array that stores the strain, boundaries points and force-load points
        self.observation_space = gym.spaces.Dict(
            {
                "strains": gym.spaces.Box(low=0.0, high=1., shape=(size_x,size_y), dtype=onp.float32),
                "boundary": gym.spaces.Box(low=0, high=1, shape=(size_x,size_y), dtype=int),
                "forces": gym.spaces.Box(low=0, high=1, shape=(size_x,size_y), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(size_x * size_y)

        
        '''
        TODO: Create the image array that we will be using for rendering, where:
            - boundary cells = 2 (red)
            - force-load cells = 3 (green)
            - removed cells = 0 (white)
            - the rest of the cells = 1 (gray)
        ''' 
        self._render_image = onp.ones((size_x, size_y))


        # The render mode we are using
        assert render_mode is None or render_mode in ["human", "rgb_array"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _coloring(self):
        add_bounded_mask = onp.zeros((self.size_x, self.size_y))
        for bounded in self.bounded_cells:
            add_bounded_mask += onp.where(self.jax_model.cell_inds_matrix == bounded, 1, 0)

        add_loaded_mask = onp.zeros((self.size_x, self.size_y))
        for loaded in self.loaded_cells:
            add_loaded_mask += onp.where(self.jax_model.cell_inds_matrix == loaded, 2, 0)

        self._render_image = onp.ones((self.size_x, self.size_y)) + add_bounded_mask + add_loaded_mask

    
    def _remove_cell_color(self, x, y):
        self._render_image[x, y] = 0

    # TODO: Implement _get_obs correctly        
    def _get_obs(self):
        self._strains, self._bounds, self._forces = self.state_tensor_check[0,:,:], self.state_tensor_check[1,:,:], self.state_tensor_check[2,:,:]
        return {"strains": self._strains,
                "boundary": self._bounds,
                "forces": self._forces}
    
    def _get_info(self):
        return self.rho_matrix

    # TODO: Debug this
    def reset(self, seed=123, options=123):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.bounded_cells, self.loaded_cells = self.jax_model.select_bounded_and_loaded_cells()
        # print(f'self_loaded cells: {self.loaded_cells}')
        # exit()
        self.max_num_step = len(self.cells) - (len(self.bounded_cells) + len(self.loaded_cells))
        
        
        self.bounded_cells = [0,5]
        self.loaded_cells = np.array([30]) #[onp.random.choice([30:35],size=1)]
        self.bounded_points, self.loaded_points = self.jax_model.select_bounded_and_loaded_points(self.bounded_cells, self.loaded_cells)

        self.dirichlet_bc = self.jax_model.set_dirichlet_bc(self.bounded_points)

        self.neumann_bc = self.jax_model.set_neumann_bc(self.loaded_points)
        self.problem = self.jax_model.problem_define(self.dirichlet_bc, self.neumann_bc)
        
        self.rho_vector = onp.copy(self.initial_rho_vector)
        self.rho_matrix = self.jax_model._state_matrix_from_array(self.rho_vector, self.size_x, self.size_y)
        self.solution = self.jax_model.problem_solve(self.problem, self.rho_vector)
        
        #self.disp_init = self.solution.reshape((self.size_x+1, self.size_y+1, -1))

        #print(f'Initial disp_init nodes: {self.disp_init}')
        # self.disp_ele_init = np.zeros((self.size_x, self.size_y, 2))
        # for i in range(self.size_x):
        #         for j in range(self.size_y):
        #             self.disp_ele_init = self.disp_ele_init.at[i,j,:].set(self.jax_model.compute_4_points_polygon_centroid(self.disp_init[i+1, j, :] , self.disp_init[i+1,j+1, :],
        #                                                                                 self.disp_init[i,j+1, :], self.disp_init[i,j, :])) 
        
        # #print(f'Initial disp_init elements: {self.disp_ele_init}')
        # initial_stiffness = self.rho_vector.reshape((self.size_x, self.size_y))
        
        #self.initial_strain_energy = (self.disp_ele_init[:,:,0].T @ initial_stiffness) @ self.disp_ele_init[:,:,0]
        #self.initial_strain_energy+= (self.disp_ele_init[:,:,1].T @ initial_stiffness) @ self.disp_ele_init[:,:,1]
        
        #print(f'Initial Solution : {self.solution}')

        self.init_SE = 0
        for elem_nb in range(self.size_x*self.size_y):
                elem_node_1 = int(elem_nb + elem_nb/self.size_x)
                elem_nodes = [elem_node_1, elem_node_1 + 1, elem_node_1 + self.size_x+1, elem_node_1 + self.size_x]
                #print(elem_nodes)
                #print(self.solution[elem_nodes,:])
                #exit(1)
                self.init_SE += (self.solution[elem_nodes,:].reshape((-1,1)).T * self.rho_vector[elem_nb]) @ self.solution[elem_nodes,:].reshape((-1,1))
        self.init_SE = onp.float(self.init_SE)
        print(f'Initial Strain Energy : {self.init_SE}')
        self.initial_von_mises = self.problem.compute_von_mises_stress(self.solution)
        self.state_tensor_DQN, self.state_tensor_check = self.jax_model.create_state_space_tensor(self.rho_vector, self.initial_von_mises, self.bounded_cells, self.loaded_cells)
        
        #print(f'DQN input tensor: {self.state_tensor_DQN}')
        print(f'Check the tensor : {self.state_tensor_check}')
        # List with the cells that we have already removed
        self.removed_cells = []

        self.current_state_tensor_DQN, self.current_state_tensor_check = self.state_tensor_DQN, self.state_tensor_check
        
        self.nb_removed_cells = 0
        self._coloring()

        observation = self._get_obs()
        info = self._get_info()

        self.special_print(0)
        return self.current_state_tensor_DQN


    def step(self, action):
        
        ## Action reperesents the cell number to remove from topology
        cell_to_be_removed = action
        reward = 0
        self.next_state_tensor_DQN = None
        if self.jax_model.check_illegal(self.rho_matrix,\
         cell_to_be_removed, self.current_state_tensor_check,\
              self.nb_removed_cells, self.max_num_step):
            reward = -1
            terminated = True
            indices = onp.argwhere(self.jax_model.cell_inds_matrix==cell_to_be_removed)
            index_x, index_y = indices[0][0], indices[0][1]
            self._remove_cell_color(index_x, index_y)
            self.special_print(f"{self.nb_removed_cells + 1} --> illegal")
        else:
            terminated = False
            if self.nb_removed_cells > 1:
                self.current_state_tensor = self.next_state_tensor_DQN

            self.nb_removed_cells += 1
            rho_vector, rho_matrix = self.jax_model.update_density(self.rho_vector,\
                 cell_to_be_removed)
            self.rho1d = rho_vector.reshape((-1,1))
            solution = self.jax_model.problem_solve(self.problem, rho_vector)
            von_mises = self.problem.compute_von_mises_stress(solution)

            self.curr_SE = 0
            for elem_nb in range(self.size_x * self.size_y):
                elem_node_1 = int(elem_nb + elem_nb/self.size_x)
                elem_nodes = [elem_node_1, elem_node_1 + 1, elem_node_1 +\
                               self.size_x+1, elem_node_1 + self.size_x]
                self.curr_SE += \
                    (solution[elem_nodes,:].reshape((-1,1)).T) @\
                          solution[elem_nodes,:].reshape((-1,1))
            self.curr_SE = onp.float(self.curr_SE)
            print()
            print(f'Current Strain Energy: {self.curr_SE}')
            print()

            reward = self.jax_model.positive_reward(self.init_SE,\
                 self.curr_SE, self.nb_removed_cells,\
                      self.size_x*self.size_y)
            self.next_state_tensor_DQN, self.next_state_tensor_check=\
            self.jax_model.create_state_space_tensor(rho_vector,\
            von_mises, self.bounded_cells, self.loaded_cells)
            
            print(f'Check the current tensor :\
                   {self.next_state_tensor_check}')

            indices =\
             onp.argwhere(self.jax_model.cell_inds_matrix==cell_to_be_removed)
            index_x, index_y = indices[0][0], indices[0][1]
            self._remove_cell_color(index_x, index_y)
            self.special_print(self.nb_removed_cells)
            self.removed_cells.append(cell_to_be_removed)
            
        return self.current_state_tensor_DQN, action, reward,\
              self.next_state_tensor_DQN, terminated
    

    def special_print(self, counter): 
        def aux(value: int):
            if value < 10:
                return (f"0{value}")
            else:
                return (f"{value}")
        index = -1    
        print(f"Step: {counter}")
        for i in range(self.size_y):
            for j in range(self.size_x):
                index += 1
                if self._render_image[i][j] == 0:
                    print(Style.BRIGHT + Back.WHITE + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # White background
                elif self._render_image[i][j] == 1:
                    print(Style.BRIGHT + Back.BLUE + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # Blue background
                elif self._render_image[i][j] == 2:
                    print(Style.BRIGHT + Back.RED + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # Red background
                elif self._render_image[i][j] == 3:
                    print(Style.BRIGHT + Back.GREEN + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # Green background
                else:
                    print(Style.BRIGHT + Back.MAGENTA + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # Magenta background
            print()
        print()
