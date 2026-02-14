"""
Experience replay buffer for off-policy RL algorithms.

Author: Jannik Rombach
"""

import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """ 
    Fixed-size buffer storing expeience tuples.
    Samples uniform random batches to break temporal correlations.
    """

    def __init__(self, capacity: int):
        """Initialize replay buffer with fixed capacity.
        
        Args:
            capacity (int): Maximum of transitions to store in buffer.
                            Oldest are drpped when capacity is exceeded (FIFO).
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store a single transistion in the buffer.

        Args:
            state (np.ndarray): Current environment observation.
            action (np.ndarray): Action taken in current state.
            reward (float): Reward received after taking action.
            next_state (np.ndarray): Resulting observation after taking action.
            done (bool): Whether the episode ended after taking action.
        """
        self.buffer.append((
            state.copy(), 
            action.copy(), 
            float(reward), 
            next_state.copy(), 
            float(done)
        ))

    def sample(self, batch_size: int) -> tuple:
        """Sample random batch of transitions for training.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: (s,a,r,s',d) as numpy arrays with shape (batch_size, ...).
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones)
        )

    def __len__(self):
        """Return current number of transitions stored in buffer."""
        return len(self.buffer)