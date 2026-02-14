"""
Neural network architectures for continuous control RL algorithms.
Implements actor (deterministic policy) and critic (Q-network) from DDPG [1][2].

Author: Jannik Rombach

References:
[1] Lillicrap et al. (2016): "Continuous control with deep reinforcement learning"
    (DDPG) - https://arxiv.org/abs/1509.02971
[2] OpenAi Spinning Up: "Deep Deterministic Policy Gradient"
    (DDPG) - https://spinningup.openai.com/en/latest/algorithms/ddpg.html
"""

import torch
import torch.nn as nn
import numpy as np

class DeterministicPolicy(nn.Module):
    """
    Deterministic policy network, state to one action.
    Maps states to continous actions via μ(s).
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        """Initialize deterministic policy network.

        Args:
            state_dim (int): Dimension of environment observation space.
            action_dim (int): Dimension of environment action space.
            max_action (float): Maximum action value for output layer scaling.
            hidden_dim (int, optional): Number of units in hidden layers. Defaults to 256.
        """
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute deterministic action for given state.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).

        Returns:
            torch.Tensor: Action tensor of shape (batch_size, action_dim)
            scaled to range [-max_action, max_action].
        """

        return self.max_action * self.net(state)
        

class QNetwork(nn.Module):
    """
    Q-value network, state-action to value.
    Maps state-action pairs to expected return via Q(s,a).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize Q-value network.

        Args:
            state_dim (int): Dimension of environment observation space.
            action_dim (int): Dimension of environment action space.
            hidden_dim (int, optional): Number of units in hidden layers. Defaults to 256.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output is scalar Q-value
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for given state-action pair.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim).
        Returns:
            torch.Tensor: Q-value tensor of shape (batch_size, 1).
        """
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)

