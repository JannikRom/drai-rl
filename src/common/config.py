"""
Central configuration management for RL training.

Author: Jannik Rombach
"""

from dataclasses import dataclass, field
from typing import Any, Dict
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
    agent_type: str
    env_name: str
    seed: int = 42

    total_timesteps: int = 100000
    batch_size: int = 256
    learning_starts: int = 10000
    replay_capacity: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    log_dir: str = "runs"

    # all agent specific params
    agent_params: Dict[str, Any] = field(default_factory=dict)

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

    
    def get(self, key:str, default=None):
        """Get config value from either known fields or agent_params.

        Args:
            key (str): Parameter name to retrieve
            default (_type_, optional): Value to return if key not found.

        Returns:
            Parameter value or defaut if not found.
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.agent_params.get(key, default)