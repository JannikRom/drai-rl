""""
Self-play trainer for the Hockey environment.

Author: Jannik Rombach
"""

from __future__ import annotations

import numpy as np
import hockey.hockey_env as hockey_env

from agents.base_agent import BaseAgent
from common.config import RLConfig
from training.standard_trainer import StandardTrainer
from training.opponent_pool import OpponentPool
from environments.environments import get_env_dims

class SelfPlayTrainer(StandardTrainer):
    """
    Extends StandardTrainer with a growing opponent pool.
    """

    def __init__(
            self,
            agent: BaseAgent,
            config: RLConfig
    ):
        super().__init__(agent, config)

        self.pool_update_interval = config.get("pool_update_interval")
        self.pool_update_check_window = config.get("pool_update_check_window")
        self.pool_update_eval_episodes = config.get("pool_update_eval_episodes")
        self.pool_update_win_rate_threshold = config.get("pool_update_win_rate_threshold")
        
        self._episodes_since_opponent_switch = 0
        self._opponent_switch_interval = 5
        
        self._episodes_since_last_pool_update = 0

        pool_max_size= config.get("pool_max_size")
        pool_strong_bot_prob = config.get("pool_strong_bot_prob")
        pool_snapshot_prob = config.get("pool_snapshot_prob")
        pool_recency_bias = config.get("pool_recency_bias")

        self.pool = OpponentPool(
            max_size = pool_max_size,
            p_strong_bot_prob=pool_strong_bot_prob,
            p_snapshot_prob = pool_snapshot_prob,
            recency_bias = pool_recency_bias
        )
        
        state_dim, action_dim, max_action = get_env_dims(self.config.env_name)
        self.pool.set_basic_opponents(
            strong_bot=hockey_env.BasicOpponent(weak=False),
            weak_bot=hockey_env.BasicOpponent(weak=True)
        )

        if self.config.get("use_fixed_opponent_pool"):
            self.pool.add_fixed_opponent_pool(state_dim, action_dim, max_action)

        if self.config.use_fixed_opponent:
            opp_type = 'sac' if self.config.agent_type.lower() == 'td3' else 'td3'
            fixed_opp = self.pool.load_fixed_opponent(opp_type, self.config, state_dim, action_dim, max_action)
            self.pool.add_fixed_opponent(fixed_opp)

        print(f"OpponentPool initialized:       {self.pool}")
        print(f"Pool update interval:           {self.pool_update_interval} episodes")
        print(f"Pool update eval episodes:      {self.pool_update_eval_episodes}")
        print(f"Pool update win-rate threshold: {self.pool_update_win_rate_threshold}")

    def train(self) -> None:
        self._print_header()
        self.logger.log_hyperparameters(self._hparams())

        opponent = self.pool.sample()
        self.env.set_opponent(opponent)

        state, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0.0
        episode_length = 0
        episode_num = 0

        for timestep in range(1, self.config.total_timesteps + 1):

            if timestep < self.config.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(state, eval_mode=False)
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_length += 1

            if timestep >= self.config.learning_starts:
                beta = self._per_beta(timestep) if self.config.get("buffer_type") == "per" else None
                losses = self.agent.train(self.buffer, self.config.batch_size, beta=beta)
                self.training_losses.append(losses)

                self.logger.log_losses(losses, timestep)
                self.logger.log_per_beta(beta, timestep)

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_num += 1
                self._episodes_since_last_pool_update += 1

                self.logger.log_episode(
                    episode_reward, episode_length, episode_num, timestep,
                    extra={
                        "SelfPlay/Winner": float(info.get("winner", 0)),
                        "SelfPlay/Pool_Size": float(len(self.pool)),
                    },
                )

                if self.logger.should_print(episode_num):
                    self.logger.log_moving_averages(
                        self.episode_rewards, self.episode_lengths, episode_num
                    )
                    w = self.config.logging.avg_window
                    avg_r = np.mean(self.episode_rewards[-w:])
                    opp_name = getattr(opponent, "_pool_name", type(opponent).__name__)
                    print(
                        f"Episode {episode_num:4d} | Step {timestep:7d} | "
                        f"Avg Reward: {avg_r:7.2f} | "
                        f"Pool: {len(self.pool)}/{self.pool.max_size} | "
                        f"Opponent: {opp_name}"
                    )

                self._maybe_update_pool(episode_num)

                # Sample new opponent after potential pool update
                self._episodes_since_opponent_switch += 1

                if self._episodes_since_opponent_switch >= self._opponent_switch_interval: 
                    opponent = self.pool.sample() 
                    self.env.set_opponent(opponent) 
                    self._episodes_since_opponent_switch = 0

                if hasattr(self.agent, 'reset_exploration'):
                    self.agent.reset_exploration()
                
                state, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            if timestep % self.config.eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=self.config.eval_episodes)
                self.logger.log_eval(eval_reward, timestep)
                print(f"  → Eval at step {timestep:7d}: {eval_reward:.2f}")

            if timestep % self.config.save_interval == 0:
                save_path = self.save_dir / f"agent_step_{timestep}.pth"
                self.agent.save(str(save_path))
                print(f"  → Saved: {save_path}")
        
        print("\n" + "=" * 60)
        final_reward = self.evaluate(num_episodes=self.config.eval_episodes)
        self.logger.log_eval(final_reward, self.config.total_timesteps, tag="Final/Eval_Reward")
        print(f"Final Evaluation: {final_reward:.2f}")
        print("=" * 60)

        self.save_results()
        self.logger.close()
        self.env.close()
    
    
    def _eval_vs_pool(self) -> float:
        """Mini round-robin tournamen against all current pool members."""
        wins, total = 0, 0
        n = self.pool_update_eval_episodes

        for opponent in self.pool.members():
            self.eval_env.set_opponent(opponent)
            for ep in range(n):
                state, _ = self.eval_env.reset(seed=ep)
                done = False

                while not done:
                    action = self.agent.select_action(state, eval_mode=True)
                    state, _, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated

                if info.get('winner', 0) == 1:
                    wins += 1
                total += 1

        self.eval_env.set_opponent(self.pool._strong_bot)
        return wins / total if total > 0 else 0.0
    

    def _make_frozen_copy(self) -> BaseAgent:
        """Creates a weight-frozen copy of the current agent for the opponent pool."""
        agent_class = type(self.agent)
        
        # Get dims from original agent
        state_dim = self.agent.state_dim
        action_dim = self.agent.action_dim
        max_action = self.agent.max_action
        
        # Create fresh agent with SAME config used for original
        frozen = agent_class(state_dim, action_dim, max_action, self.config)
        
        # Copy only inference weights (actor/policy)
        if hasattr(self.agent, 'actor'):  # SAC
            frozen.actor.load_state_dict(self.agent.actor.state_dict())
            frozen.actor.eval()
            for param in frozen.actor.parameters():
                param.requires_grad_(False)
        elif hasattr(self.agent, 'policy'):  # TD3
            frozen.policy.load_state_dict(self.agent.policy.state_dict())
            frozen.policy.eval()
            for param in frozen.policy.parameters():
                param.requires_grad_(False)
        else:
            raise ValueError(f"Agent {agent_class.__name__} must have 'actor' or 'policy' attribute")
        
        frozen._pool_name = f"{agent_class.__name__}_frozen"
        return frozen



    def _maybe_update_pool(self, episode_num: int) -> None:
        """
        Add current agent snapshot to the pool if:
          1. Enough episodes have passed since the last update.
          2. Enough reward history is available.
          3. Win-rate vs all pool members >= threshold.
        """
        if self._episodes_since_last_pool_update < self.pool_update_interval:
            return
        if len(self.episode_rewards) < self.pool_update_check_window:
            return

        win_rate = self._eval_vs_pool()
        avg_reward = float(np.mean(self.episode_rewards[-self.pool_update_check_window:]))
        self._episodes_since_last_pool_update = 0

        if win_rate >= self.pool_update_win_rate_threshold:
            frozen = self._make_frozen_copy()
            name = f"{self.agent.__class__.__name__}_ep{episode_num}_wr{win_rate:.2f}"
            self.pool.add(frozen, name=name)
            self.logger.log_pool(len(self.pool), episode_num, win_rate, avg_reward)
            print(
                f"  → Pool updated ep {episode_num}: "
                f"win_rate={win_rate:.1%}  avg_reward={avg_reward:.2f}  "
                f"pool_size={len(self.pool)}"
            )
        else:
            print(
                f"  → Pool update skipped ep {episode_num}: "
                f"win_rate={win_rate:.1%} < {self.pool_update_win_rate_threshold:.1%}"
            )