"""
Policy Iteration Algorithm 

SARSA

[[0,1,0,0],
 [0,0,1,0],
 [0,0,0,1],
 [1,0,0,2]]

 Starting point is (0,0)
 Terminal state is (3,3)

 Reward function -> 0 or 1 shown in the maze

 Task : Agent should find the best policy to reach the terminal state with the highest reward

 discount factor : 0.
 learning rate : 0.1
"""

import numpy as np
import time
from tqdm import tqdm
      
class SARSA:
   
    def __init__(self, states, actions, actions_map, discount_factor, learning_rate, Q_table):
        self.states = states
        self.actions = actions 
        self.actions_map = actions_map
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.Q_table = Q_table
        self.NUM_ACTIONS = len(actions)
        self.ACTION_INDEXES = np.arange(self.NUM_ACTIONS)
        self.GRID_SIZE = (4,4)
        self.TERMINAL_STATE = (3,3)
      
    def choose_action(self, pos, t):
        """
        Choose the action with the epsilon-greedy policy
        """
        epsilon = 1.0 / float(t) if t > 0 else 1.0
        r, c = pos
        # exploration
        if np.random.rand() < epsilon :
            return np.random.choice(self.ACTION_INDEXES)
        # exploitation
        else:
            return np.argmax(self.Q_table[r, c, :])

    def step(self, pos, action_index):
        """
        Perform the selected action in the current state
        Goal : Observe the next state and reward
        Return : next_pos, reward

        """

        # get the true action to the environment
        action = self.actions[action_index]

        # perform the action
        next_pos = (pos[0] + action[0], pos[1] + action[1])

        # check if the next_pos is out of bounds
        if not (0 <= next_pos[0] < self.GRID_SIZE[0] and 0 <= next_pos[1] < self.GRID_SIZE[1]):
            # Hitting the wall
            next_pos = pos
            
        
        # check if the next_pos is a terminal state
        is_terminal = (next_pos == self.TERMINAL_STATE)

        reward = self.states[next_pos[0]][next_pos[1]]      
        return next_pos, reward, is_terminal
    
    def update_Q_table(self, pos, old_action_index, new_pos, reward, next_action_index, is_terminal):
        # the old Q value
    
        old_Q = self.Q_table[pos[0], pos[1], old_action_index]

        if not is_terminal:
            TD_target = reward + self.discount_factor * self.Q_table[new_pos[0], new_pos[1], next_action_index]
        else:
            TD_target = reward

        TD_error = TD_target - old_Q
        
        # update the Q-table
        self.Q_table[pos[0], pos[1], old_action_index] += self.learning_rate * TD_error

        

def train(states, actions, actions_map, discount_factor, learning_rate, Q_table, num_episodes, max_step, start_pos=(0,0)) :
    """
    """
    # SARSA instance
    sarsa = SARSA(states, actions, actions_map, discount_factor, learning_rate, Q_table)
    
    episode_progress = tqdm(range(num_episodes), desc="Training Episodes", unit="episode")

    for episode in episode_progress:
        
        t = 1
        episode_start_time = time.time()
        is_terminal = False
        current_pos = start_pos
        action_index = sarsa.choose_action(current_pos, t)
        
        step_count = 1

        while not is_terminal and step_count < max_step:
            
            next_pos, reward, is_terminal = sarsa.step(current_pos, action_index)
            # choose the next action if not terminal
            step_count += 1
            if not is_terminal:
                next_action_index = sarsa.choose_action(next_pos, t)
            else:
                next_action_index = 0
                
            # update the Q-table 
            sarsa.update_Q_table(current_pos, action_index, next_pos, reward, next_action_index, is_terminal)
            
            # update the current position
            current_pos = next_pos
            action_index = next_action_index

            # update the timestamp
            t += 1
        
        episode_time = time.time() - episode_start_time
        
        # Update progress bar description with episode info
        episode_progress.set_postfix({
            'Steps': step_count - 1,
            'Time': f'{episode_time:.2f}s',
            'Avg Reward': f'{reward:.2f}'
        })
        
        
    return sarsa.Q_table
        
if __name__ == "__main__":
    # initial parameters
    discount_factor = 0.9
    t = 0 # timestamp
    alpha = 0.1
    epsilon = 1 

    states = np.array([[0,1,0,1],
    [1,0,1,0],
    [0,1,1,0],
    [1,0,1,5]])
    max_step = 10
    actions = [(-1,0), (1,0), (0,-1), (0,1)] # up, down, left, right
    actions_map = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}

    Q_table = np.zeros((4,4,4)) # (row, col, action)
    Trained_Q_table = train(states, actions, actions_map, discount_factor, alpha, Q_table, 5000, max_step, start_pos=(0,0))
    
    np.set_printoptions(precision=4, suppress=True)
    print(Trained_Q_table)

