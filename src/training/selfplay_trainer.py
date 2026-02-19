""""
Self-play trainer for the Hockey environment.

Author: Jannik Rombach
"""

from __future__ import annotations

import numpy as np
import hockey.hockey_env as hockey_env

from training.trainer import Trainer
from training.opponent_pool import OpponentPool
from agents.base_agent import BaseAgent
from common.config import RLConfig

class SelfPlayTrainer(Trainer):
    """
    Trainer that samples opponents form a grwoing pool of past agent snapshots.
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
        """Self play training loop."""
        print("=" * 60)
        print(f"Experiment:  {self.config.experiment_name}")
        print(f"Agent:       {self.agent.__class__.__name__}")
        print(f"Mode:        self-play")
        print(f"Total steps: {self.config.total_timesteps:,}")
        print(f"Device:      {self.agent.device}")
        print(f"Seed:        {self.config.seed}")
        print("=" * 60)

        self._log_hyperparameters()

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
                if self.config.get("buffer_type") == "per":
                    progress = (timestep - self.config.learning_starts) / (
                        (self.config.total_timesteps - self.config.learning_starts)
                        * self.config.get("per_annealing_pct")
                    )
                    beta = min(1.0, self.per_beta_start + progress * (1.0 - self.per_beta_start))
                else:
                    beta = 1.0

                losses = self.agent.train(self.buffer, self.config.batch_size, beta=beta)

                self.training_losses.append(losses)

                if timestep % 1000 == 0:
                    self.writer.add_scalar('Loss/Critic', losses['critic_loss'], timestep)
                    if losses['actor_loss'] > 0:
                        self.writer.add_scalar('Loss/Actor', losses['actor_loss'], timestep)

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_num += 1
                self._episodes_since_last_pool_update += 1

                self.writer.add_scalar('Train/Episode_Reward', episode_reward, episode_num)
                self.writer.add_scalar('Train/Episode_Length', episode_length, episode_num)
                self.writer.add_scalar('Train/Pool_Size', len(self.pool), episode_num)
                self.writer.add_scalar('Train/Winner', info.get('winner', 0), episode_num)

                if episode_num % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    self.writer.add_scalar('Train/Average_Reward_10', avg_reward, episode_num)

                    print(
                        f"Episode {episode_num:4d} | Step {timestep:7d} | "
                        f"Avg Reward (10ep): {avg_reward:7.2f} | "
                        f"Pool: {len(self.pool)}/{self.pool.max_size} | "
                        f"Opponent: {getattr(opponent, '_pool_name', type(opponent).__name__)}"
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
                self.writer.add_scalar('Eval/Reward', eval_reward, timestep)
                print(f"  → Eval at step {timestep:7d}: {eval_reward:.2f}")

            if timestep % self.config.save_interval == 0:
                save_path = self.save_dir / f"agent_step_{timestep}.pth"
                self.agent.save(str(save_path))
                print(f"  → Saved: {save_path}")
        
        print("\n" + "=" * 60)
        final_reward = self.evaluate(num_episodes=self.config.eval_episodes)
        self.writer.add_scalar('Eval/Final_Reward', final_reward, self.config.total_timesteps)
        print(f"Final Evaluation: {final_reward:.2f}")
        print("=" * 60)

        self.save_results()
        self.writer.close()
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
        Add current agent to pool if:
        1. Enough episodes have passed since last update
        2. Enough reward history is available
        3. Win-rate vs all pool members >= pool_update_win_rate_threshold
        """
        if self._episodes_since_last_pool_update < self.pool_update_interval:
            return
        if len(self.episode_rewards) < self.pool_update_check_window:
            return
        
        win_rate = self._eval_vs_pool()

        if win_rate >= self.pool_update_win_rate_threshold:
            avg_reward = np.mean(self.episode_rewards[-self.pool_update_check_window:])
            name = f"{self.agent.__class__.__name__}_ep{episode_num}_wr{win_rate:.2f}"

            self.pool.add(self.agent, name=name)
            self._episodes_since_last_pool_update = 0
            print(
                f"  → Pool updated at episode {episode_num}: "
                f"win_rate={win_rate:.1%} avg_reward={avg_reward:.2f} | "
                f"Pool size: {len(self.pool)}"
            )
            self.writer.add_scalar('Train/Pool_Update_WinRate', win_rate, episode_num)
            self.writer.add_scalar('Train/Pool_Update_Reward',  avg_reward, episode_num)
        else:
            print(
                f"  → Pool update skipped at episode {episode_num}: "
                f"win_rate={win_rate:.1%} < {self.pool_update_win_rate_threshold:.1%}"
            )
            self._episodes_since_last_pool_update = 0