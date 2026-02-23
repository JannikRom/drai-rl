"""
Opponent pool for self-play training.

Maintains a pool of past agent snapshots sampled ranodmly during training.
Random pool sampling prevents overfitting.

Author: Jannik Rombach
"""

from __future__ import annotations

import random
import numpy as np
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from common.config import RLConfig
import os
import glob

THIS_DIR = os.path.dirname(__file__)

SAC_CHECKPOINT = os.path.join(THIS_DIR, "strong_sac.pth")
SAC_CONFIG = os.path.join(THIS_DIR, "strong_sac.yaml")
TD3_CHECKPOINT = os.path.join(THIS_DIR, "strong_td3.pth")
TD3_CONFIG = os.path.join(THIS_DIR, "strong_td3.yaml")

FIXED_POOL_DIR = os.path.join(THIS_DIR, "fixed_opponents")


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
        self._fixed_opponent_pool: list = []
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
        
        if random.random() < self.p_snapshot_prob and self._rotated: 
            n = len(self._rotated)   
            weights = np.linspace(1.0, self.recency_bias, n) 
            weights /= weights.sum()   
            return random.choices(self._rotated, weights=weights, k=1)[0]   
        
        choices = [self._weak_bot]   
        if hasattr(self, '_fixed_opponent') and self._fixed_opponent is not None: 
            choices.append(self._fixed_opponent)
        if hasattr(self, '_fixed_opponent_pool') and self._fixed_opponent_pool:
            choices.extend(self._fixed_opponent_pool)
        return random.choice(choices)

        
    def members(self) -> list:
        members = [self._strong_bot, self._weak_bot]
        if hasattr(self, '_fixed_opponent') and self._fixed_opponent:
            members.append(self._fixed_opponent)
        if hasattr(self, '_fixed_opponent_pool'):
            members.extend(self._fixed_opponent_pool)
        members += self._rotated
        return members

    def __len__(self) -> int:
        return len(self._rotated)
    
    def __repr__(self):
        fixed_str = f", fixed_pool={len(getattr(self, '_fixed_opponent_pool', []))}" if hasattr(self, '_fixed_opponent_pool') else ""
        fixed_single = f", fixed={self._fixed_opponent is not None}" if hasattr(self, '_fixed_opponent') else ""
        return (
            f"OpponentPool("
            f"snapshots={len(self._rotated)}/{self.max_size}, "
            f"p_strong={self.p_strong_bot_prob}, "
            f"p_snapshot={self.p_snapshot_prob}, "
            f"recency_bias={self.recency_bias}{fixed_str}{fixed_single})"
        )
    
    def load_fixed_opponent(self, agent_type: str, config: RLConfig, state_dim: int, action_dim: int, max_action: float):
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


    def add_fixed_opponent_pool(self, state_dim: int, action_dim: int, max_action: float):
            """Load all fixed opponents from hardcoded fixed_opponents/ dir."""
            if not os.path.exists(FIXED_POOL_DIR):
                print(f"No fixed_opponents/ dir at {FIXED_POOL_DIR}")
                self._fixed_opponent_pool = []
                return
            
            self._fixed_opponent_pool = []
            checkpoint_pattern = os.path.join(FIXED_POOL_DIR, "*.pth")
            for pth_path in glob.glob(checkpoint_pattern):
                yaml_path = pth_path.replace('.pth', '.yaml')
                if not os.path.exists(yaml_path):
                    print(f"Skipping {pth_path}: no matching .yaml")
                    continue
                name = os.path.splitext(os.path.basename(pth_path))[0]
                opponent = self._load_fixed_from_paths(pth_path, yaml_path, name, state_dim, action_dim, max_action)
                if opponent:
                    self._fixed_opponent_pool.append(opponent)
            print(f"Loaded {len(self._fixed_opponent_pool)} fixed opponents from {FIXED_POOL_DIR}")

    def _load_fixed_from_paths(self, checkpoint: str, config_path: str, name: str, 
                                state_dim: int, action_dim: int, max_action: float):
        agent_type = 'sac' if 'sac' in name.lower() else 'td3'
        if agent_type.lower() == 'sac':
            AgentClass = SACAgent
        elif agent_type.lower() == 'td3':
            AgentClass = TD3Agent
        else:
            print(f"Skipping {name}: unknown type")
            return None
        
        temp_config = RLConfig.from_yaml(config_path)
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
        
        opponent._pool_name = f"{agent_type}_fixed_{name}"
        print(f"Loaded FROZEN {agent_type} '{name}' from {checkpoint}")
        return opponent




