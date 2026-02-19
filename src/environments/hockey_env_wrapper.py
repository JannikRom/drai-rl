"""
Gymnasium wrapper for the Hockey environment.

Handles opponent management, reward shaping and single-agent interface for the base Trainer. 
Supports self-play via set_opponent() for SelfPlayTrainer.

Author: Jannik Rombach, Adriano Polzer
"""


import gymnasium as gym
import numpy as np
import hockey.hockey_env as hockey_env

class HockeyEnvWrapper(gym.Env):

    def __init__(self, mode='NORMAL', opponent='weak', reward_shaping=None):
        """Init of HockeyEnvWrapper

        Args:
            mode (str, optional): 'NORMAL', 'TRAIN_SHOOTING', or 'TRAIN_DEFENSE'
            opponent (str, optional): 'weak' or 'strong'
            reward_shaping (_type_, optional): Dict with weights for auxiliary rewards
        """
        
        self.env = gym.make('Hockey-v0', mode=getattr(hockey_env.Mode, mode))
        self.opponent_type = opponent

        # Setup initial opponent
        if opponent == 'weak':
            self.opponent = hockey_env.BasicOpponent(weak=True)
        elif opponent == 'strong':
            self.opponent = hockey_env.BasicOpponent(weak=False)
        else:
            self.opponent = None
        
        # Reward shaping weights
        default_shaping = {
            'closeness_to_puck': 0.0,
            'touch_puck': 0.0,
            'puck_direction': 0.0
        }
        if reward_shaping:
            default_shaping.update(reward_shaping)
        self.reward_shaping = default_shaping


        # Spaces (agent controls player 1, 4-dim action)
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)

    def set_opponent(self, opponent) -> None:
        "Inject an opponent from outside for self-play training."
        self.opponent = opponent

    def reset (self, seed = None, options = None):
        obs, info = self.env.reset(seed = seed, options = options)
        return obs, info
    
    def step(self, action):
        # Get opponent action
        if self.opponent is not None:
            obs_agent2 = self.env.unwrapped.obs_agent_two()
            opponent_action = self._get_action(self.opponent, obs_agent2)
        else:
            opponent_action = np.zeros(4)
        
        # Combine actions
        full_action = np.concatenate([action, opponent_action])

        # Step environment
        obs, reward, done, truncated, info = self.env.step(full_action)

        # Apply reward shaping
        shaped_reward = reward
        shaped_reward += self.reward_shaping['closeness_to_puck'] * info.get('reward_closeness_to_puck', 0)
        shaped_reward += self.reward_shaping['touch_puck'] * info.get('reward_touch_puck', 0)
        shaped_reward += self.reward_shaping['puck_direction'] * info.get('reward_puck_direction', 0)

        return obs, shaped_reward, done, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


    def _get_action(self, opponent, obs: np.ndarray) -> np.ndarray:
        """Unified action interface for BasicOpponent and trained agents."""
        if opponent is None:
            return np.zeros(4)
        if hasattr(opponent, 'select_action'):
            return opponent.select_action(obs, eval_mode=True)
        elif hasattr(opponent, 'act'):
            return opponent.act(obs)
        else:
            raise TypeError(f"Unsupported opponent type: {type(opponent)}")

