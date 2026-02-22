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
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from common.config import RLConfig



SAC_CHECKPOINT = "environments/strong_sac.pth" 
SAC_CONFIG = "environments/strong_sac.yaml" 

TD3_CHECKPOINT = "environments/strong_td3.pth" 
TD3_CONFIG = "environments/strong_td3.yaml" 

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
        self._fixed_opponent = None
        self._rotated: list = [] # agent snapshots, FIFO

    def set_basic_opponents(self, strong_bot, weak_bot):
        """Set the permanent basic opponents."""
        self._strong_bot = strong_bot
        self._weak_bot = weak_bot

        self._strong_bot._pool_name = "BasicOpponent_strong"
        self._weak_bot._pool_name   = "BasicOpponent_weak"

    def add_fixed_opponent(self, opponent):
        """Add fixed opponent alongside basics (sampled like weak bot)."""
        self._fixed_opponent = opponent

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
                # Sample weak or fixed equally
                return random.choice([self._weak_bot, self._fixed_opponent]) if self._fixed_opponent else self._weak_bot
            n_rotated = len(self._rotated)
            weights = np.linspace(1.0, self.recency_bias, n_rotated)
            weights = weights / weights.sum()
            return random.choices(self._rotated, weights=weights, k=1)[0]
        else:
            # Sample weak or fixed equally
            return random.choice([self._weak_bot, self._fixed_opponent]) if self._fixed_opponent else self._weak_bot
        
    def members(self) -> list:
        members = [self._strong_bot, self._weak_bot]
        if self._fixed_opponent:
            members.append(self._fixed_opponent)
        members += self._rotated
        return members

    def __len__(self) -> int:
        return len(self._rotated)
    
    def __repr__(self) -> str:
        fixed_str = f", fixed={self._fixed_opponent is not None}" if hasattr(self, '_fixed_opponent') else ""
        return (
            f"OpponentPool("
            f"snapshots={len(self._rotated)}/{self.max_size}, "
            f"p_strong={self.p_strong_bot_prob}, "
            f"p_snapshot={self.p_snapshot_prob}, "
            f"recency_bias={self.recency_bias}{fixed_str})"
        )
    
def load_fixed_opponent(agent_type: str, config: RLConfig, state_dim: int, action_dim: int, max_action: float):
    """Load frozen SAC or TD3 using original training configs."""
    if agent_type.lower() == 'sac':
        checkpoint = SAC_CHECKPOINT
        AgentClass = SACAgent
        temp_config = RLConfig.from_yaml(SAC_CONFIG)
    elif agent_type.lower() == 'td3':
        checkpoint = TD3_CHECKPOINT
        AgentClass = TD3Agent
        temp_config = RLConfig.from_yaml(TD3_CONFIG)
    else:
        raise ValueError(f"Unknown opponent type: {agent_type}")
    
    temp_config.agent_type = agent_type 
    
    opponent = AgentClass(state_dim, action_dim, max_action, temp_config)
    opponent.load(checkpoint)
    
    # FULL FREEZE
    for attr in ['actor', 'policy', 'critic_1', 'critic_2']:
        if hasattr(opponent, attr):
            module = getattr(opponent, attr)
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
    
    opponent._pool_name = f"{agent_type}_fixed_pretrained"
    print(f"Loaded FROZEN {agent_type} from {checkpoint} (using {SAC_CONFIG or TD3_CONFIG})")
    return opponent




