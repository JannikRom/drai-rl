"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

Implements three key improvements over DDPG:
1. Clipped double-Q learning to reduce overestimation bias
2. Target policy smoothing to prevent overfitting to deterministic targets
3. Delayed policy updates to let critics converge before actor updates

Author: Jannik Rombach

References:
[1] Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods"
    https://arxiv.org/abs/1802.09477
    Algorithm 1: TD3 with twin critics, target smoothing, delayed updates
[2] OpenAI Spinning Up: "Twin Delayed DDPG"
    https://spinningup.openai.com/en/latest/algorithms/td3.html 
"""

import torch
import torch.nn as nn
import torch.optim
import numpy as np
from common.networks import DeterministicPolicy, QNetwork
from common.replay_buffer import ReplayBuffer

class TD3Agent:

    def __init__(self, state_dim: int, action_dim: int, max_action: float, config: dict):
        """
        Initialize TD3 agent.

        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Dimension of action space.
            max_action (float): Maximum absolute value for actions.
            config (dict): Dictionary with hyperparameters.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = max_action
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.policy_delay = config.get("policy_delay", 2)
        self.noise_std = config.get("noise_std", 0.1)
        self.noise_clip = config.get("noise_clip", 0.5)
        hidden_dim = config.get("hidden_dim", 256)

        # Networks
        self.policy = DeterministicPolicy(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.policy_target = DeterministicPolicy(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.critic_1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("actor_lr", 3e-4))
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) +
                                                  list(self.critic_2.parameters()), lr=config.get("critic_lr", 3e-4)
                                                  )
        
        # Training state
        self.total_updates = 0

    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select action from current policy with optional exploration noise.

        Args:
            state (np.ndarray): current environment state.
            eval_mode (bool, optional): If True, return deterministic action without noise.

        Returns:
            np.ndarray: Action array clipped to [-max_action, max_action].
        """
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy(state_tensor).cpu().numpy().flatten()
        
        if not eval_mode:
            # Add exploration noise during training
            noise = np.random.normal(0, self.noise_std * self.max_action, size=self.action_dim)
            action = np.clip(action + noise , -self.max_action, self.max_action)

        return action
    

    def train(self, replay_buffer: ReplayBuffer, batch_size: int) -> dict:
        """
        Perform one TD3 training step on a sampled batch.

        Args:
            replay_buffer (ReplayBuffer): Experiece replay buffer.
            batch_size (int): Number of tranistions to sample.

        Returns:
            dict: Dictionary with critic_loss and actor_loss.
        """
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute targes Q-value with targes policy smoothing 
        with torch.no_grad():
            # Target action with smoothing noise
            target_noise = torch.randn_like(actions) * self.noise_std
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = self.policy_target(next_states) + target_noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)

            # Clipped double-Q: min of two target Q-values
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # TD target
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Update critics
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        critic_1_loss = nn.MSELoss()(current_q1, target_q)
        critic_2_loss = nn.MSELoss()(current_q2, target_q)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy update
        actor_loss = None
        if self.total_updates % self.policy_delay == 0:
            # Actor loss: maximze Q-value
            actor_loss = -self.critic_1(states, self.policy(states)).mean()

            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            # Soft update targets
            self._soft_update(self.policy, self.policy_target)
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)

        self.total_updates += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update target networks parameters using Polyak averaging.

        θ_target = τ * θ_source + (1 - τ) * θ_target
        """

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        """Save current policy and critic networks to disk."""
        torch.save({
            "policy": self.policy.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict()
        }, path)

    def load(self, path: str):
        """Load networks and sync targets."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])

        # Sync targets
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())