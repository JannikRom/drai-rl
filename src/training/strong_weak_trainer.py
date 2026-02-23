"""
Strong/Weak opponent trainer for Hockey environment.

Trains against strong opponent (80%) and weak opponent (20%).
Each opponent is used for exactly 5 consecutive episodes before switching.

Author: Jannik Rombach, Adriano Polzer
"""

from __future__ import annotations

import numpy as np
import hockey.hockey_env as hockey_env

from agents.base_agent import BaseAgent
from common.config import RLConfig
from training.standard_trainer import StandardTrainer


class StrongWeakTrainer(StandardTrainer):
    """
    Trains against strong (80%) and weak (20%) scripted opponents.
    Each opponent type is used for exactly 5 consecutive episodes.
    """
    
    def __init__(self, agent: BaseAgent, config: RLConfig):
        super().__init__(agent, config)
        
        # Create fixed opponents
        self.strong_opponent = hockey_env.BasicOpponent(weak=False)
        self.weak_opponent = hockey_env.BasicOpponent(weak=True)
        
        # Training schedule
        self.episodes_per_opponent = 5
        self.strong_prob = 0.8
        
        # Track state
        self._current_opponent = None
        self._episodes_with_current_opponent = 0
        
        print(f"Strong/Weak trainer: strong={self.strong_prob:.1%}, "
              f"episodes_per_opponent={self.episodes_per_opponent}")
    
    def train(self) -> None:
        self._print_header()
        self.logger.log_hyperparameters(self._hparams())
        
        state, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0.0
        episode_length = 0
        episode_num = 0
        
        self._sample_opponent()  
        
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
            
            # Training step
            if timestep >= self.config.learning_starts:
                beta = self._per_beta(timestep)
                losses = self.agent.train(self.buffer, self.config.batch_size, beta=beta)
                self.training_losses.append(losses)
                self.logger.log_losses(losses, timestep)
                self.logger.log_per_beta(beta, timestep)
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_num += 1
                self._episodes_with_current_opponent += 1
                
                self.logger.log_episode(
                    episode_reward, episode_length, episode_num, timestep,
                    extra={
                        "Opponent/Is_Strong": float(self._current_opponent == self.strong_opponent),
                        "Opponent/Remaining_Episodes": self.episodes_per_opponent - self._episodes_with_current_opponent
                    }
                )
                
                if self.logger.should_print(episode_num):
                    self.logger.log_moving_averages(
                        self.episode_rewards, self.episode_lengths, episode_num
                    )
                    w = self.config.logging.avg_window
                    avg_r = np.mean(self.episode_rewards[-w:])
                    opp_name = "strong" if self._current_opponent == self.strong_opponent else "weak"
                    print(
                        f"Episode {episode_num:4d} | Step {timestep:7d} | "
                        f"Avg Reward: {avg_r:7.2f} | "
                        f"Opponent: {opp_name} ({self._episodes_with_current_opponent}/{self.episodes_per_opponent})"
                    )
                
                if self._episodes_with_current_opponent >= self.episodes_per_opponent:
                    self._sample_opponent()
                
                # Reset exploration and environment
                if hasattr(self.agent, "reset_exploration"):
                    self.agent.reset_exploration()
                
                state, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            
            if timestep % self.config.eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=self.config.eval_episodes)
                self.logger.log_eval(eval_reward, timestep)
                print(f"  → Eval at step {timestep:7d}: {eval_reward:.2f}")
            
            if timestep % self.config.save_interval == 0:
                ckpt = self.save_dir / f"agent_step_{timestep}.pth"
                self.agent.save(str(ckpt))
                print(f"  → Checkpoint: {ckpt}")


        
        # Final evaluation
        print("\n" + "=" * 60)
        final_reward = self.evaluate(num_episodes=self.config.eval_episodes)
        self.logger.log_eval(final_reward, self.config.total_timesteps, tag="Eval/Final_Reward")
        print(f"Final Evaluation: {final_reward:.2f}")
        print("=" * 60)
        
        self.save_results()
        self.logger.close()
        self.env.close()
        self.eval_env.close()

    def evaluate(self, num_episodes: int = 5) -> float:
        """
        OVERRIDE: Evaluate using only terminal win/loss rewards (+10/0/-10).
        Ignores shaping rewards during evaluation.
        """
        eval_rewards = []
        
        # Use strong opponent for consistency
        self.eval_env.set_opponent(self.strong_opponent)
        
        for ep in range(num_episodes):
            state, _ = self.eval_env.reset(seed=ep)
            episode_reward = 0.0
            done = False
            
            while not done:
                action = self.agent.select_action(state, eval_mode=True)
                state, raw_reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                # only win loss
                if done:
                    winner = info.get('winner', 0)  # 1=player1 win, -1=player2 win, 0=draw
                    episode_reward = 10.0 if winner == 1 else (0.0 if winner == 0 else -10.0)
            
            eval_rewards.append(episode_reward)
        
        return float(np.mean(eval_rewards))



    
    
    def _sample_opponent(self):
        """Sample opponent with 80% strong, 20% weak."""
        if np.random.random() < self.strong_prob:
            self._current_opponent = self.strong_opponent
            opp_name = "BasicOpponent_strong"
        else:
            self._current_opponent = self.weak_opponent
            opp_name = "BasicOpponent_weak"
        
        self.env.set_opponent(self._current_opponent)
        self._episodes_with_current_opponent = 0

    def _hparams(self) -> dict:
        """Extended hyperparameters with trainer-specific params."""
        base = super()._hparams()
        base.update({
            "trainer_type": "StrongWeakTrainer",
            "strong_prob": self.strong_prob,
            "episodes_per_opponent": self.episodes_per_opponent
        })
        return base
    
    def _print_header(self) -> None:
        """Extended header with trainer info."""
        super()._print_header()
        print(f"Trainer:     StrongWeakTrainer")
        print(f"Strong prob: {self.strong_prob:.1%}")
        print(f"Ep/opponent: {self.episodes_per_opponent}")
        print("=" * 60)
