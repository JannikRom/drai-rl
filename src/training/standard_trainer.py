"""
Standard (singe-agent) trainer

Author: Jannik Rombach, Adriano Polzer
"""

from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path

from agents.base_agent import BaseAgent
from common.config import RLConfig
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from environments.environments import make_env
from training.logger import  TrainingLogger

class StandardTrainer:

    def __init__(self, agent: BaseAgent, config: RLConfig):
        """
        Initialize trainer for RL agent.
        
        Args:
            agent: RL agent implementing BaseAgent interface
            config: RLConfig object with all hyperparameters
        """
        self.agent = agent
        self.config = config

        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create environment
        self.env = make_env(config.env_name, config)

        # Initialize buffer
        self.buffer = self._build_buffer()

        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

        # Create save directory
        self.save_dir = Path(config.log_dir) / config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Logger
        self.logger = TrainingLogger(self.save_dir / "tensorboard", config.logging)
        
        print(f"Save directory: {self.save_dir}")
        

    def train(self):
        """Main training loop."""

        self._print_header()
        self.logger.log_hyperparameters(self._hparams())

        state, _ = self.env.reset(seed=self.config.seed)
        episode_reward = 0
        episode_length = 0
        episode_num = 0

        for timestep in range(1, self.config.total_timesteps + 1):

            # Select action
            if timestep < self.config.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(state, eval_mode=False)

            # Execute action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1

            # Train agent
            if timestep >= self.config.learning_starts:
                beta = self._per_beta(timestep)
                losses = self.agent.train(self.buffer, self.config.batch_size, beta=beta)

                self.training_losses.append(losses)
                self.logger.log_losses(losses, timestep)
                self.logger.log_per_beta(beta, timestep)
            
            # Episode finished
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_num += 1

                self.logger.log_episode(episode_reward, episode_length, episode_num, timestep)

                if self.logger.should_print(episode_num):
                    self.logger.log_moving_averages(
                        self.episode_rewards, self.episode_lengths, episode_num
                    )
                    w = self.config.logging.avg_window
                    avg_r = np.mean(self.episode_rewards[-w:])
                    avg_l = np.mean(self.episode_lengths[-w:])
                    print(
                        f"Episode {episode_num:4d} | Step {timestep:7d} | "
                        f"Avg Reward: {avg_r:7.2f} | Avg Length: {avg_l:5.1f}"
                    )

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

        print("\n" + "=" * 60)
        final_reward = self.evaluate(num_episodes=self.config.eval_episodes)
        self.logger.log_eval(final_reward, self.config.total_timesteps, tag="Eval/Final_Reward")
        print(f"Final Evaluation: {final_reward:.2f}")
        print("=" * 60)

        self.save_results()
        self.logger.close()
        self.env.close()
                   

    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate agent performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Average episode reward
        """
        eval_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, eval_mode=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)

        return float(np.mean(eval_rewards))
    
    def save_results(self):
        """Save model and training metrics."""
       
        model_path = self.save_dir / "agent_final.pth"
        self.agent.save(str(model_path))
        print(f"Saved final model to {model_path}")

        if self.config.logging.save_numpy_arrays:
            np.save(self.save_dir / "episode_rewards.npy", self.episode_rewards)
            np.save(self.save_dir / "episode_lengths.npy", self.episode_lengths)

            if self.training_losses:
                    loss_history = {
                        key: np.array([step[key] for step in self.training_losses if key in step])
                        for key in self.training_losses[0].keys()
                    }
                    np.savez(self.save_dir / "training_losses.npz", **loss_history)
        
        if self.config.logging.save_config_yaml:
            self._save_config(self.save_dir / "config_used.yaml")
        
        if self.config.logging.save_reward_plot:
            self._plot_rewards()

    def _build_buffer(self) -> ReplayBuffer | PrioritizedReplayBuffer:
            if self.config.get("buffer_type") == "per":
                self.per_beta_start = float(self.config.get("per_beta_start"))
                return PrioritizedReplayBuffer(
                    capacity=self.config.replay_capacity,
                    alpha=float(self.config.get("per_alpha")),
                    epsilon=float(self.config.get("per_epsilon")),
                )
            return ReplayBuffer(self.config.replay_capacity)

    def _per_beta(self, timestep: int) -> float:
        if self.config.get("buffer_type") != "per":
            return 1.0
        progress = (timestep - self.config.learning_starts) / (
            (self.config.total_timesteps - self.config.learning_starts)
            * self.config.get("per_annealing_pct")
        )
        return min(1.0, self.per_beta_start + progress * (1.0 - self.per_beta_start))

    def _hparams(self) -> dict:
            base = {
                "agent_type": self.config.agent_type,
                "env_name": self.config.env_name,
                "gamma": self.config.gamma,
                "tau": self.config.tau,
                "batch_size": self.config.batch_size,
                "learning_starts": self.config.learning_starts,
                "seed": self.config.seed,
            }
            base.update({
                k: v for k, v in self.config.agent_params.items()
                if isinstance(v, (int, float, str, bool))
            })
            return base

    def _print_header(self) -> None:
            print("=" * 60)
            print(f"Experiment:  {self.config.experiment_name}")
            print(f"Agent:       {self.agent.__class__.__name__}")
            print(f"Environment: {self.config.env_name}")
            print(f"Mode:        {self.config.training_mode}")
            if self.config.mode:
                print(f"Env mode:    {self.config.mode}")
            if self.config.opponent:
                print(f"Opponent:    {self.config.opponent}")
            print(f"Total steps: {self.config.total_timesteps:,}")
            print(f"Device:      {self.agent.device}")
            print(f"Seed:        {self.config.seed}")
            print("=" * 60)

    def _save_config(self, path: Path) -> None:
        config_dict = {
            "experiment_name": self.config.experiment_name,
            "seed": self.config.seed,
            "env_name": self.config.env_name,
            "mode": self.config.mode,
            "opponent": self.config.opponent,
            "reward_shaping": self.config.reward_shaping,
            "agent_type": self.config.agent_type,
            "gamma": self.config.gamma,
            "tau": self.config.tau,
            "total_timesteps": self.config.total_timesteps,
            "learning_starts": self.config.learning_starts,
            "batch_size": self.config.batch_size,
            "replay_capacity": self.config.replay_capacity,
            "log_dir": self.config.log_dir,
            "save_interval": self.config.save_interval,
            "eval_interval": self.config.eval_interval,
            "eval_episodes": self.config.eval_episodes,
            **self.config.agent_params,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Saved config: {path}")

    def _plot_rewards(self) -> None:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.episode_rewards, alpha=0.3, label="Episode Reward")

            window = min(50, max(1, len(self.episode_rewards) // 10))
            if len(self.episode_rewards) >= window and window > 1:
                moving_avg = np.convolve(
                    self.episode_rewards, np.ones(window) / window, mode="valid"
                )
                ax.plot(
                    range(window - 1, len(self.episode_rewards)),
                    moving_avg, linewidth=2, label=f"{window}-ep Moving Avg",
                )

            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.set_title(self.config.experiment_name)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.savefig(self.save_dir / "training_rewards.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved: {self.save_dir / 'training_rewards.png'}")
        except ImportError:
            print("Matplotlib not available, skipping plot.")

