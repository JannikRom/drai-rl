"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

Implements three key improvements over DDPG:
1. Clipped double-Q learning to reduce overestimation bias
2. Target policy smoothing to prevent overfitting to deterministic targets
3. Delayed policy updates to let critics converge before actor updates

Author: Jannik Rombach

References:
[1] Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods"
[2] OpenAI Spinning Up: "Twin Delayed DDPG"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from common.config import RLConfig
from common.networks import DeterministicPolicy, QNetwork
from common.replay_buffer import ReplayBuffer
from agents.base_agent import BaseAgent
from common.noise import get_noise


class TD3Agent(BaseAgent):
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, config: RLConfig):
        """
        Initialize TD3 agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum absolute value for actions
            config: Dictionary with hyperparameters
        """
        super().__init__(state_dim, action_dim, max_action, config)
        
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.gamma = config.get("gamma")
        self.tau = config.get("tau")
        self.policy_delay = config.get("policy_delay")
        self.policy_noise = config.get("policy_noise")
        self.noise_clip = config.get("noise_clip")
        
        actor_hidden_sizes = config.get("actor_hidden_sizes")
        critic_hidden_sizes = config.get("critic_hidden_sizes")
        
        self.exploration_noise = get_noise(config=config)
        
        # Networks
        self.policy = DeterministicPolicy(state_dim, action_dim, max_action, actor_hidden_sizes).to(self.device)
        self.policy_target = DeterministicPolicy(state_dim, action_dim, max_action, actor_hidden_sizes).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        
        self.critic_1 = QNetwork(state_dim, action_dim, critic_hidden_sizes).to(self.device)
        self.critic_1_target = QNetwork(state_dim, action_dim, critic_hidden_sizes).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = QNetwork(state_dim, action_dim, critic_hidden_sizes).to(self.device)
        self.critic_2_target = QNetwork(state_dim, action_dim, critic_hidden_sizes).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.get("actor_lr")
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=config.get("critic_lr")
        )
        
        # Training state
        self.total_updates = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action with optional exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.policy(state_tensor).cpu().numpy().flatten()
        
        if not eval_mode:
            noise = self.exploration_noise.sample()
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def reset_exploration(self):
        """ Resets noise process at episode start (important for correlated noise) """
        self.exploration_noise.reset()
    
    def train(self, replay_buffer: ReplayBuffer, batch_size: int, beta: float = 1.0) -> dict:
        """Perform one TD3 training step."""
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta=beta)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Compute target Q-value with target policy smoothing
        with torch.no_grad():
            target_noise = torch.randn_like(actions) * self.policy_noise * self.max_action
            target_noise = torch.clamp(
                target_noise, -self.noise_clip * self.max_action, self.noise_clip * self.max_action
            )
            next_actions = self.policy_target(next_states) + target_noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)
            
            # Clipped double-Q
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Update critics
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        td_error1 = current_q1 - target_q
        td_error2 = current_q2 - target_q
        critic_loss = (weights * (td_error1**2 + td_error2**2)).mean() # MSE loss weighted by importance sampling weights

        #critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update priorities in replay buffer if using PER
        if indices is not None:
            with torch.no_grad():
                priorities = (torch.abs(td_error1) + torch.abs(td_error2)).cpu().numpy().flatten() / 2.0
                replay_buffer.update_priorities(indices, priorities)
                
        # Delayed policy update
        actor_loss = None
        if self.total_updates % self.policy_delay == 0:
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
            "actor_loss": actor_loss.item() if actor_loss is not None else None
        }
    
    def save(self, path: str, timestep: int = 0) -> None:
        torch.save({
            "timestep": timestep,
            "total_updates": self.total_updates,
            "policy": self.policy.state_dict(),
            "policy_target": self.policy_target.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str, weights_only: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(ckpt["policy"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])

        # Always restore targets from checkpoint if available
        if "policy_target" in ckpt:
            self.policy_target.load_state_dict(ckpt["policy_target"])
            self.critic_1_target.load_state_dict(ckpt["critic_1_target"])
            self.critic_2_target.load_state_dict(ckpt["critic_2_target"])
        else:
            # Legacy checkpoint — sync targets from main networks
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.critic_1_target.load_state_dict(self.critic_1.state_dict())
            self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        if not weights_only:
            self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            self.total_updates = ckpt.get("total_updates", 0)
            print(f"Resumed from timestep {ckpt.get('timestep', 'unknown')} "
                  f"({self.total_updates} updates)")
        else:
            print(f"Loaded weights from timestep {ckpt.get('timestep', 'unknown')} "
                  f"(fine-tune mode)")
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target networks using Polyak averaging."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
