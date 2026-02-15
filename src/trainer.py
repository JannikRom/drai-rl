import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from agents.base_agent import BaseAgent
from common.replay_buffer import ReplayBuffer
from common.config import RLConfig
from environments.hockey_env_wrapper import HockeyEnvWrapper


class Trainer:

    def __init__(self, agent: BaseAgent, config: RLConfig):
        """
        Initialize trainer for RL agent.
        
        Args:
            agent: RL agent implementing BaseAgent interface
            config: Configuration object with hyperparameters
        """
        self.agent = agent
        self.config = config

        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create environment
        if config.env_name == 'Hockey-v0':
            self.env = HockeyEnvWrapper(
                mode=getattr(config, 'mode', 'NORMAL'),
                opponent=getattr(config, 'opponent', 'random'),
                reward_shaping=getattr(config, 'reward_shaping', {})
            )
        else:
            self.env = gym.make(config.env_name)

        # Initialize buffer
        self.buffer = ReplayBuffer(config.replay_capacity)

        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

        # Create save directory
        agent_name = agent.__class__.__name__.lower().replace("agent", "")
        self.save_dir = Path(config.log_dir) / f"{agent_name}_{config.env_name}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """Main training loop."""
        print("=" * 60)
        print(f"Training {self.agent.__class__.__name__} on {self.config.env_name}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Device: {self.agent.device}")
        print("=" * 60)

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
                losses = self.agent.train(self.buffer, self.config.batch_size)
                self.training_losses.append(losses)
            
            # Episode finished
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_num += 1

                # Log progress
                if episode_num % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(f"Episode {episode_num:4d} | Timestep {timestep:6d} | "
                          f"Avg Reward: {avg_reward:7.2f} | Avg Length: {avg_length:5.1f}")
                    
                # Reset environment
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

            # Periodic evaluation
            if timestep % 50000 == 0:
                eval_reward = self.evaluate(num_episodes=5)
                print(f"  → Evaluation at timestep {timestep}: {eval_reward:.2f}")
        
        # Final evaluation
        print("\n" + "=" * 60)
        final_reward = self.evaluate(num_episodes=10)
        print(f"Final Evaluation: {final_reward:.2f}")
        print("=" * 60)
        
        self.save_results()
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

        return np.mean(eval_rewards)
    
    def save_results(self):
        """Save model and training metrics."""
        # Save model
        model_path = self.save_dir / "agent.pth"
        self.agent.save(str(model_path))
        print(f"Saved model to {model_path}")

        # Save training data
        np.save(self.save_dir / "episode_rewards.npy", self.episode_rewards)
        np.save(self.save_dir / "episode_lengths.npy", self.episode_lengths)

        # Plot rewards
        self._plot_rewards()

    def _plot_rewards(self):
        """Plot training rewards."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
            
            # Moving average
            window = 10
            if len(self.episode_rewards) >= window:
                moving_avg = np.convolve(self.episode_rewards, 
                                        np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(self.episode_rewards)), 
                        moving_avg, linewidth=2, label=f'{window}-Episode Moving Avg')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'{self.agent.__class__.__name__} on {self.config.env_name}')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plot_path = self.save_dir / "training_rewards.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {plot_path}")
        except ImportError:
            print("Matplotlib not available, skipping plot")
