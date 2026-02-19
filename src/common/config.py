"""
Central configuration management for RL training.

Author: Jannik Rombach
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import yaml
from pathlib import Path

@dataclass
class RLConfig:
    """
    Central configuration for RL training loaded from YAML files.

    Supports inheritance: base.yaml provided defaults, 
    experiment-specific YAML can override and add new params.

    Unknown params are stored in agent_params for algorithm-specific settings.
    """
    # Experiment
    experiment_name: str
    seed: int

    # Environment
    env_name: str 
    mode: Optional[str] = None # Hockey: 'NORMAL', 'TRAIN_SHOOTING' or 'TRAIN_DEFENSE'
    opponent: Optional[str] = None # Hockey: 'weak' or 'strong'
    reward_shaping: Dict[str, float] = field(default_factory=dict)

    # Agent
    agent_type: str
    gamma: float
    tau: float

    # Buffer parameters
    buffer_type: str = 'rb' # 'rb' for ReplayBuffer, 'per' for PrioritizedReplayBuffer
    per_alpha: float = 0.6
    per_epsilon: float = 1e-6
    per_beta_start: float = 0.4
    per_annealing_pct: float = 0.8
    # All other agent-specific parameters
    agent_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training
    total_timesteps: int
    learning_starts: int
    batch_size: int
    replay_capacity: int

    #Logging
    log_dir: str = "./logs"
    save_interval: int
    eval_interval: int
    eval_episodes: int


    @classmethod
    def from_yaml(cls, config_path: str) -> 'RLConfig':
        """Load conftig from YAML.

        Args:
            config_path (str): Path to experiment YAML file.

        Returns:
            RLConfig: RLConfig instance.
        """

        config_path = Path(config_path)
        base_path = config_path.parent / "base.yaml"

        # Load base config
        config_dict = {}
        if base_path.exists():
            with open(base_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}

        # Load specific config and override base
        with open(config_path, 'r') as f:
            specific_config = yaml.safe_load(f) or {}
            config_dict.update(specific_config)

        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        extra_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        
        return cls(**known_params, agent_params=extra_params)

    
    def get(self, key:str):
        """Get config value from either known fields or agent_params.

        Args:
            key (str): Parameter name to retrieve

        Returns:
            Parameter value
        Raises:
            KeyErros: If key not found.
        """
        if hasattr(self, key):
            return getattr(self, key)
        
        if key in self.agent_params:
            return self.agent_params[key]
        
        raise KeyError(
            f"Required config key '{key}' not found. "
            f"Available agent_params: {list(self.agent_params.keys())}"
        )