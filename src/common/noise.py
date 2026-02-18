"""
Action noise generators for RL exploration in continous control.

- White Gaussian noise (TD3/DDPG standard)
- Pink noise via AR(1) process for temporally correlated exploration [1]

Author: Jannik Rombach

References:
[1] Eberhard et al. (2023): "Pink Noise is All You Need: Colored Noise Exploration 
    in Deep Reinforcement Learning" (ICLR) - 
    https://openreview.net/forum?id=hQ9V5QN27eS
"""

import numpy as np
from common.config import RLConfig
from common.environments import get_env_dims

class WhiteNoise:
    """
    Standard uncorrelated Gaussian noise (TD3/DDPG default).
    """
    def __init__(self, action_dim: int, scale: float = 1.0):
        self.action_dim = action_dim
        self.scale = scale

    def reset(self):
        pass  # stateless

    def sample(self) -> np.ndarray:
        return np.random.normal(0, self.scale, size=self.action_dim).astype(np.float32)
    
class PinkNoise:
    """
    Temporally correlated noise via AR(1) prcoess.
    """
    def __init__(self, action_dim: int, beta: float = 0.9, scale: float = 1.0):
        self.action_dim = action_dim
        self.scale = scale
        self.beta = beta
        self.state = np.zeros(action_dim, dtype=np.float32)

    def reset(self):
        self.state[:] = 0.0

    def sample(self) -> np.ndarray:
        eta = np.random.randn(self.action_dim).astype(np.float32)
        self.state = (self.beta * self.state + 
                      np.sqrt(1.0 - self.beta**2) * eta)

        return self.state * self.scale
    
def get_noise(config: RLConfig) -> WhiteNoise | PinkNoise:

    noise_type = config.get("noise_type")
    state_dim, action_dim, max_action = get_env_dims(config.env_name)
    scale = config.get("noise_scale")
    

    if noise_type == "white":
        return WhiteNoise(action_dim, scale=scale)
    elif noise_type == "pink":
        beta = config.get("pink_beta")
        return PinkNoise(
            action_dim, 
            scale=scale, 
            beta=beta
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    