import numpy as np
import random

class GridEnvironment:
    def __init__(self):
        self.grid_size = 3
        self.reset()

    def reset(self):
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        self.goal_pos = self.random_empty_square()
        self.grid[self.goal_pos] = 2  # Goal is represented by 2
        
        self.trap_pos = self.random_empty_square()
        self.grid[self.trap_pos] = -2  # Trap is represented by -2

        #self.barrier_pos = self.random_empty_square()

        #self.grid[self.barrier_pos] = -1  # Barrier is represented by -1

        self.agent_pos = self.random_empty_square()

        return self.get_flattened_state()
    
    def random_empty_square(self):
        while True:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if self.grid[pos] == 0:
                return pos

    #def get_state(self):
        ## Flatten the grid and append the agent's position
        ##return (self.grid.flatten())
        #return np.concatenate((self.grid.flatten(), [self.agent_pos[0], self.agent_pos[1]]))/(self.grid_size-1)

    def get_flattened_state(self):
        # Create a copy of the grid and mark the agent's position
        grid_copy = self.grid.copy()
        grid_copy[self.agent_pos] = 3  # Agent is represented by 3
        # Flatten the grid to create the state vector
        return grid_copy.flatten().reshape(1, -1)


    def step(self, action):
        reward = 0
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_pos = (self.agent_pos[0] + moves[action][0], self.agent_pos[1] + moves[action][1])

        # Check if new position is out of bounds
        if new_pos[0] < 0 or new_pos[0] >= self.grid_size or new_pos[1] < 0 or new_pos[1] >= self.grid_size:
            reward -= 5  # collision penalty
            new_pos = self.agent_pos  # Stay in place

        # Check if new position is a barrier
        # if new_pos == self.barrier_pos:
        #     new_pos = self.agent_pos  # Stay in place

        # Move agent
        self.grid[self.agent_pos] = 0
        self.grid[new_pos] = 1
        self.agent_pos = new_pos

        # Calculate reward
        reward -= 1  # Default move penalty
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
        elif self.agent_pos == self.trap_pos:
            reward = -100
            done = True

        return self.get_flattened_state(), reward, done
    
    def render(self):
        env = self.grid.copy()
        env[self.agent_pos] = 1  # Agent is represented by 1
        print(env)
