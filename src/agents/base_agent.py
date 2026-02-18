"""
Base agent interface for reinforcement learning algorithms.

Ensures consistent API across all agent implementations (TD3, SAC, PPO, etc.)
for use with the unified Trainer class.

Author: Jannik Rombach
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
from common.config import RLConfig


class BaseAgent(ABC):
    """Abstract base class for RL agents."""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, config: RLConfig):
        """
        Initialize agent.
        
        Args:
            state_dim: Dimension of observation space
            action_dim: Dimension of action space
            max_action: Maximum absolute value for actions
            config: Hyperparameter dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Select best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    @abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select action given current state.
        
        Args:
            state: Current environment state
            eval_mode: If True, use deterministic/greedy policy
            
        Returns:
            Action array
        """
        pass
    
    @abstractmethod
    def train(self, replay_buffer, batch_size: int) -> dict:
        """
        Perform one training step.
        
        Args:
            replay_buffer: Experience replay buffer
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary of loss metrics for logging
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent parameters to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent parameters from disk."""
        pass
