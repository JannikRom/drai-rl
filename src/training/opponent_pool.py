"""
Opponent pool for self-play training.

Maintains a pool of past agent snapshots sampled ranodmly during training.
Random pool sampling prevents overfitting.

Author: Jannik Rombach
"""

from __future__ import annotations

import copy
import random
import numpy as np

class OpponentPool:
    """
    Pool of past agent snapshots for self-play training.
    """

    def __init__(self, max_size: int = 20, p_strong_bot_prob: float = 0.8 ,p_snapshot_prob: float = 0.5, recency_bias: float = 2.0):
        self.max_size = max_size
        self.p_strong_bot_prob = p_strong_bot_prob
        self.p_snapshot_prob = p_snapshot_prob
        self.recency_bias = recency_bias
        
        self._strong_bot = None # permanent strong basic opponent
        self._weak_bot = None # permanent weak basic opponent
        self._rotated: list = [] # agent snapshots, FIFO

    def set_basic_opponents(self, strong_bot, weak_bot):
        """Set the permanent basic opponents."""
        self._strong_bot = strong_bot
        self._weak_bot = weak_bot

        self._strong_bot._pool_name = "BasicOpponent_strong"
        self._weak_bot._pool_name   = "BasicOpponent_weak"

    def add(self, agent, name: str = None, permanent: bool = False) -> None:
        """Copy agent weights to new instance and add to pool."""
        agent_class = type(agent)
        
        # Get dims from agent
        state_dim = agent.state_dim
        action_dim = agent.action_dim
        max_action = agent.max_action

        snapshot = agent_class(state_dim, action_dim, max_action, agent.config)
        
        # Copy only inference weights
        if hasattr(agent, 'actor'):  # SAC
            snapshot.actor.load_state_dict(agent.actor.state_dict())
            snapshot.actor.eval()
            for param in snapshot.actor.parameters():
                param.requires_grad_(False)
        elif hasattr(agent, 'policy'):  # TD3
            snapshot.policy.load_state_dict(agent.policy.state_dict())
            snapshot.policy.eval()
            for param in snapshot.policy.parameters():
                param.requires_grad_(False)
        
        snapshot._pool_name = name or agent_class.__name__
        
        if len(self._rotated) >= self.max_size:
            self._rotated.pop(0)
        
        self._rotated.append(snapshot)



    def sample(self) -> object:
        """
        Sample an opponent with recency-weighted probability over rotated entries, and uniform probability for permanent entries.
        
        p_strong_bot_prob.8: strong basic opponent

        (1 - p_strong_bot_prob) * p_snapshot_prob: snapshot from rotating pool (recency-weighted)
        (1 - p_strong_bot_prob) * (1 - p_snapshot_prob): weak basic opponent
        """
        if random.random() < self.p_strong_bot_prob:
            return self._strong_bot
 
        if random.random() < self.p_snapshot_prob:
            if not self._rotated:
                return self._weak_bot
            n_rotated = len(self._rotated)
            weights = np.linspace(1.0, self.recency_bias, n_rotated)
            weights = weights / weights.sum()
            return random.choices(self._rotated, weights=weights, k=1)[0]
        
        else:
            return self._weak_bot 
    
    def members(self) -> list:
        return [self._strong_bot, self._weak_bot] + self._rotated
    
    def __len__(self) -> int:
        return len(self._rotated)
    
    def __repr__(self) -> str:
        return (
            f"OpponentPool("
            f"snapshots={len(self._rotated)}/{self.max_size}, "
            f"p_strong={self.p_strong_bot_prob}, "
            f"p_snapshot={self.p_snapshot_prob}, "
            f"recency_bias={self.recency_bias})"
        )

