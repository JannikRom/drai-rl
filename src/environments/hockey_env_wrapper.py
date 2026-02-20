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

    def __init__(self, mode='NORMAL', opponent='weak', reward_shaping=None, render_mode=None):
        """Init of HockeyEnvWrapper"""
        self.render_mode = render_mode
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
            'puck_direction': 0.0,
            'time_penalty': 0.0,
            'defensive_distance': 0.0
        }
        if reward_shaping:
            default_shaping.update(reward_shaping)
        self.reward_shaping = default_shaping
        self.step_counter = 0
        self.max_steps = 250

        # Spaces (agent controls player 1, 4-dim action)
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)

    def set_opponent(self, opponent) -> None:
        "Inject an opponent from outside for self-play training."
        self.opponent = opponent

    def reset (self, seed = None, options = None):
        self.step_counter = 0
        obs, info = self.env.reset(seed = seed, options = options)
        return obs, info
    
    def step(self, action):
        # Get opponent action
        self.step_counter += 1
        if self.opponent is not None:
            obs_agent2 = self.env.unwrapped.obs_agent_two()
            opponent_action = self._get_action(self.opponent, obs_agent2)
        else:
            opponent_action = np.zeros(4)
        
        # Combine actions
        full_action = np.concatenate([action, opponent_action])

        # Step environment
        obs, reward, done, truncated, info = self.env.step(full_action)

        shaped_reward = self._calculate_custom_rewards(obs, reward, info)

        return obs, shaped_reward, done, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

    def _calculate_custom_rewards(self, obs, reward, info):
        # identical with environment
        SCALE = 60.0 
        VIEWPORT_W = 600
        W = VIEWPORT_W / SCALE
        CENTER_X = W / 2  

        VIEWPORT_GOAL_X = W / 2 - 245 / SCALE - 10 / SCALE
        OBS_GOAL_X = VIEWPORT_GOAL_X - CENTER_X

        player_pos = obs[0:2] 
        puck_x = obs[12]
        puck_vel = obs[14:16]
        puck_speed = np.sqrt(puck_vel[0]**2 + puck_vel[1]**2)

        # Time Penalty 
        time_penalty = 0
        if self.step_counter > (self.max_steps * 0.1):
            if puck_x < 0 and puck_speed < 0.1 and info.get('reward_touch_puck') == 0:
                time_penalty = -0.05

        # Defensive Distance Reward 
        goal_center = np.array([OBS_GOAL_X, 0.0]) 
        dist_to_goal = np.linalg.norm(player_pos - goal_center)
        
        # linear reward based on distance to goal, only if puck is opponents side
        reward_defensive = 0
        if puck_x > 0:
            reward_defensive =  (1.0 / (1.0 + dist_to_goal))

        # Shaping
        shaping_closeness = self.reward_shaping.get('closeness_to_puck') * info.get('reward_closeness_to_puck', 0)
        shaping_touch = self.reward_shaping.get('touch_puck') * info.get('reward_touch_puck', 0)
        shaping_direction = self.reward_shaping.get('puck_direction') * info.get('reward_puck_direction', 0)

        shaping_time = self.reward_shaping.get('time_penalty') * time_penalty
        shaping_defensive = self.reward_shaping.get('defensive_distance') * reward_defensive
        total_reward = (reward + 
                        shaping_closeness + 
                        shaping_touch + 
                        shaping_direction + 
                        shaping_time + 
                        shaping_defensive)
        
        return total_reward

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

