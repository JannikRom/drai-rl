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
        
        self._episodes_since_last_pool_update = 0

        pool_max_size= config.get("pool_max_size")

        # Seed pool with bots basic opponents
        self.pool = OpponentPool(max_size = pool_max_size)
        self.pool.add(hockey_env.BasicOpponent(weak=True), name="BasicOpponent_weak")
        self.pool.add(hockey_env.BasicOpponent(weak=False), name="BasicOpponent_strong")

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

                opponent = self.pool.sample()
                self.env.set_opponent(opponent)

                if hasattr(self.agent, 'reset_exploration'):
                    self.agent.reset_exploration()
                
                state, _ = self.env.reset(seed=self.config.seed + episode_num)
                episode_reward = 0.0
                episode_length = 0

            if timestep % self.config.eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=self.config.eval_episodes)
                self.logger.log_eval(eval_reward, timestep)
                print(f"  → Eval at step {timestep:7d}: {eval_reward:.2f}")
                self.env.set_opponent(opponent)  # reset to current pool opponent after eval

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

 
    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluation always against the same opponent: BasicOpponent(weak=False)"""
        self.env.set_opponent(hockey_env.BasicOpponent(weak=False))
        rewards = []

        for ep in range(num_episodes):
            state, _ = self.env.reset(seed=ep)
            episode_reward = 0.0
            done = False

            while not done:
                action = self.agent.select_action(state, eval_mode=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            rewards.append(episode_reward)
        
        return float(np.mean(rewards))
    
    
    def _eval_vs_pool(self) -> float:
        """Mini round-robin tournamen against all current pool members."""
        wins, total = 0, 0
        n = self.pool_update_eval_episodes

        for opponent in self.pool.members():
            self.env.set_opponent(opponent)
            for ep in range(n):
                state, _ = self.env.reset(seed=ep)
                done = False

                while not done:
                    action = self.agent.select_action(state, eval_mode=True)
                    state, _, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                if info.get('winner', 0) == 1:
                    wins += 1
                total += 1

        return wins / total if total > 0 else 0.0

        

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
            name = f"{self.agent.__class__.__name__}_ep{episode_num}_wr{win_rate:.2f}"
            self.pool.add(self.agent, name=name)
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