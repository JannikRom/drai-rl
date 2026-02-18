"""
Experience and prioritized replay buffer for off-policy RL algorithms.

Author: Jannik Rombach, Adriano Polzer

[1] Schaul et al. (2015): "Prioritized Experience Replay"
    - https://arxiv.org/abs/1511.05952
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

    def sample(self, batch_size: int, beta: float = 0.1) -> tuple:
        """Sample random batch of transitions for training.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: (s,a,r,s',d) as numpy arrays with shape (batch_size, ...).
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            None, 
            weights

        )

    def __len__(self):
        """Return current number of transitions stored in buffer."""
        return self.size

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha=0.6, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0 

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, data)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)
        is_weights /= is_weights.max() 

        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), 
                idxs, np.array(is_weights, dtype=np.float32))

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
    
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left]
                parent = right
        
        return parent, self.tree[parent], self.data[parent - self.capacity + 1]

    @property
    def total_priority(self):
        return self.tree[0] 