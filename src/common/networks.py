"""
Neural network architectures for continuous control RL algorithms.
Implements actor (deterministic and stochastic policy) and critic (Q-network) from DDPG [1][2] and SAC [3][4].

Author: Jannik Rombach, Adriano Polzer

References:
[1] Lillicrap et al. (2016): "Continuous control with deep reinforcement learning"
    (DDPG) - https://arxiv.org/abs/1509.02971
[2] OpenAi Spinning Up: "Deep Deterministic Policy Gradient"
    (DDPG) - https://spinningup.openai.com/en/latest/algorithms/ddpg.html
[3] Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    (SAC) - https://arxiv.org/abs/1801.01290
[4] OpenAi Spinning Up: "Soft Actor-Critic"
    (SAC) - https://spinningup.openai.com/en/latest/algorithms/sac.html 
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
        
class StochasticPolicy(nn.Module):
    """
    Stochastic policy network for SAC
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, 
                 hidden_dim: int = 256):
        """Initialize stochastic policy network.

        Args:
            state_dim (int): Dimension of environment observation space.
            action_dim (int): Dimension of environment action space.
            max_action (float): Maximum action value for output scaling.
            hidden_dim (int, optional): Number of units in hidden layers. Defaults to 256.
        """
        super().__init__()
        self.max_action = max_action

        # Shared feature extraction layers
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Separate output heads for mean and log standard deviation
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log standard deviation for given state.

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).

        Returns:
            [torch.Tensor, torch.Tensor]: Mean and log_std tensors, 
            each of shape (batch_size, action_dim).
        """
        features = self.backbone(state)
        return self.mu_head(features), self.log_std_head(features)

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy distribution using reparameterization trick

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).

        Returns:
            [torch.Tensor, torch.Tensor]: 
                - Action tensor of shape (batch_size, action_dim) scaled to [-max_action, max_action].
                - Log probability tensor of shape (batch_size, 1).
        """
        mu, log_std = self.forward(state)
        log_std = torch.clamp(log_std, min=-20, max=2) 
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample() # reparameterization trick
        y = torch.tanh(x)
        action = self.max_action * y
        
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.max_action * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for given state (for evaluation/deployment).

        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).
            deterministic (bool, optional): If True, return mean action. Defaults to False.

        Returns:
            torch.Tensor: Action tensor of shape (batch_size, action_dim).
        """
        with torch.no_grad():
            if deterministic:
                mu, _ = self.forward(state)
                return self.max_action * torch.tanh(mu)
            else:
                action, _ = self.sample(state)
                return action

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

