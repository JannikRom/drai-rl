"""
Experience replay buffer for off-policy RL algorithms.

Author: Jannik Rombach
"""

import numpy as np

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
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

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

        # Initialize arrays on first push
        if self.states is None:
            self.states = np.zeros((self.capacity, *state.shape), dtype=np.float32)
            self.actions = np.zeros((self.capacity, *action.shape), dtype=np.float32)
            self.rewards = np.zeros(self.capacity, dtype=np.float32)
            self.next_states = np.zeros((self.capacity, *state.shape), dtype=np.float32)
            self.dones = np.zeros(self.capacity, dtype=np.float32)
        
        # Store at current position (circular buffer)
        idx = self.position % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        
        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """Sample random batch of transitions for training.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: (s,a,r,s',d) as numpy arrays with shape (batch_size, ...).
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        """Return current number of transitions stored in buffer."""
        return self.size