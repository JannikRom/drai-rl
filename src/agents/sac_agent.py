"""
Soft Actor-Critic (SAC) agent.

Key features:
1. Maximizes both expected reward and policy entropy to encourage exploration and robustness
2. Learns a Gaussian distribution over actions, providing inherent exploration without the need for manual noise processes (Stochastic Policy)
3. Clipped Double-Q Learning to mitigate the overestimation bias 
4. Automatic Entropy Tuning, ensuring the policy maintains a target entropy

Author: Adriano Polzer

References:
[1] Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
[2] Haarnoja et al. (2019): "Soft Actor-Critic Algorithms and Applications"
[3] OpenAI Spinning Up: "Soft Actor-Critic"
"""

import torch
import torch.nn as nn
import numpy as np
from common.networks import StochasticPolicy, QNetwork
from common.replay_buffer import ReplayBuffer
from agents.base_agent import BaseAgent


class SACAgent(BaseAgent):
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, config: dict):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum absolute value for actions
            config: Dictionary with hyperparameters
        """
        super().__init__(state_dim, action_dim, max_action, config)
        
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        hidden_dim = config.get("hidden_dim", 256)

        self.inital_alpha = config.get("alpha", 0.2)          
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(self.inital_alpha, device=self.device).log().detach().requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.get("actor_lr", 3e-4))
        self.alpha = self.log_alpha.exp()

        # Networks
        self.actor = StochasticPolicy(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        
        self.critic_1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.get("actor_lr", 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=config.get("critic_lr", 3e-4)
        )
        
        # Training state
        self.total_updates = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """ Select action from policy """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor.get_action(state_tensor, deterministic=eval_mode).cpu().numpy().flatten()
        return action
    
    def train(self, replay_buffer: ReplayBuffer, batch_size: int) -> dict:
        """Perform one SAC training step."""
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Target Q-values using clipped double-Q learning
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # TD target
            target_q = rewards + self.gamma * (1 - dones) * (target_q - self.alpha.detach() * next_log_probs)
        
        # Update critics
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        new_actions, log_probs = self.actor.sample(states)
        
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha loss for automatic entropy tuning
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Soft update targets
        self._soft_update(self.critic_1, self.critic_1_target)
        self._soft_update(self.critic_2, self.critic_2_target)
        
        self.total_updates += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_log_prob": log_probs.mean().item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target networks using Polyak averaging."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        """Save current policy and critic networks."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path: str):
        """Load networks and sync targets."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()

        # Sync targets
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())