"""
Environment dimensions, wrappers, and factories.
"""

from typing import Tuple, Dict
import gymnasium as gym
from common.config import RLConfig
from environments.hockey_env_wrapper import HockeyEnvWrapper


ENV_DIMS: Dict[str, Tuple[int, int, float]] = {
    'Hockey-v0': (18, 4, 1.0),
    'Pendulum-v1': (3, 1, 1.0),
    'LunarLanderContinuous-v3': (8, 2, 1.0),
}

def get_env_dims(env_name: str) -> Tuple[int, int, float]:
    """Get (state_dim, action_dim, max_action)."""
    if env_name not in ENV_DIMS:
        raise KeyError(f"Unknown env: {env_name}")
    return ENV_DIMS[env_name]

def make_env(env_name: str, config: RLConfig) -> gym.Env:
    """
    Factory for configured environments.
    """
    if env_name == 'Hockey-v0':
        return HockeyEnvWrapper(
            mode=config.get('mode'),
            opponent=config.get('opponent'),
            reward_shaping=config.get('reward_shaping'),
        )   
    else:
        return gym.make(env_name, env_name)

