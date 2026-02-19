"""
Universal trainer for RL agents with TensorBoard logging.

Author: Jannik Rombach
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from agents.base_agent import BaseAgent
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from common.config import RLConfig
from environments.environments import make_env

class Trainer:

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
        if config.get("buffer_type") == "per":
            self.buffer = PrioritizedReplayBuffer(
                capacity=config.replay_capacity,
                alpha=float(config.get("per_alpha")),
                epsilon=float(config.get("per_epsilon"))
            )
            self.per_beta_start = config.get("per_beta_start")
        else:
            self.buffer = ReplayBuffer(config.replay_capacity)

        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

        # Create save directory
        self.save_dir = Path(config.log_dir) / config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))
        
        print(f"Save directory: {self.save_dir}")
        print(f"TensorBoard logs: {self.save_dir / 'tensorboard'}")
        print(f"Start TensorBoard with: tensorboard --logdir={self.save_dir / 'tensorboard'}")

    def train(self):
        """Main training loop."""
        print("=" * 60)
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Environment: {self.config.env_name}")
        if self.config.mode:
            print(f"Mode: {self.config.mode}")
        if self.config.opponent:
            print(f"Opponent: {self.config.opponent}")
        print(f"Total timesteps: {self.config.total_timesteps:,}")
        print(f"Device: {self.agent.device}")
        print(f"Seed: {self.config.seed}")
        print("=" * 60)

        # Log hyperparameters to TensorBoard
        self._log_hyperparameters()

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
                
                # Log losses to TensorBoard
                if timestep % 100 == 0:
                    self.writer.add_scalar('Loss/Critic', losses['critic_loss'], timestep)
                    if 'actor_loss' in losses and losses['actor_loss'] is not None:
                        self.writer.add_scalar('Loss/Actor', losses['actor_loss'], timestep)
                    if 'alpha' in losses:
                        self.writer.add_scalar('SAC/Alpha', losses['alpha'], timestep)
                        self.writer.add_scalar('SAC/Alpha_Loss', losses['alpha_loss'], timestep)
                        self.writer.add_scalar('SAC/Mean_Log_Prob', losses['mean_log_prob'], timestep)
                    if self.config.get("buffer_type") == "per":
                        self.writer.add_scalar('PER/Beta', beta, timestep)
            
            # Episode finished
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_num += 1

                # Log to TensorBoard
                self.writer.add_scalar('Train/Episode_Reward', episode_reward, episode_num)
                self.writer.add_scalar('Train/Episode_Length', episode_length, episode_num)
                self.writer.add_scalar('Train/Episode_Reward_vs_Timestep', episode_reward, timestep)

                # Log progress
                if episode_num % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    
                    # Log moving averages to TensorBoard
                    self.writer.add_scalar('Train/Avg_Reward_10ep', avg_reward, episode_num)
                    self.writer.add_scalar('Train/Avg_Length_10ep', avg_length, episode_num)
                    
                    print(f"Episode {episode_num:4d} | Timestep {timestep:7d} | "
                          f"Avg Reward: {avg_reward:7.2f} | Avg Length: {avg_length:5.1f}")
                
                # Exploration Reset
                if hasattr(self.agent, 'reset_exploration'):
                    self.agent.reset_exploration()
                    
                # Reset environment
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

            # Periodic evaluation
            if timestep % self.config.eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=self.config.eval_episodes)
                self.writer.add_scalar('Eval/Reward', eval_reward, timestep)
                print(f"  → Evaluation at timestep {timestep:7d}: {eval_reward:.2f}")
            
            # Periodic saving
            if timestep % self.config.save_interval == 0:
                save_path = self.save_dir / f"agent_step_{timestep}.pth"
                self.agent.save(str(save_path))
                print(f"  → Saved checkpoint: {save_path}")
        
        # Final evaluation
        print("\n" + "=" * 60)
        final_reward = self.evaluate(num_episodes=self.config.eval_episodes)
        self.writer.add_scalar('Eval/Final_Reward', final_reward, self.config.total_timesteps)
        print(f"Final Evaluation: {final_reward:.2f}")
        print("=" * 60)
        
        self.save_results()
        self.writer.close()
        self.env.close()

    def _log_hyperparameters(self):
        """Log hyperparameters to TensorBoard."""
        hparams = {
            'agent_type': self.config.agent_type,
            'gamma': self.config.gamma,
            'tau': self.config.tau,
            'batch_size': self.config.batch_size,
            'learning_starts': self.config.learning_starts,
            'seed': self.config.seed,
            **{k: v for k, v in self.config.agent_params.items() 
               if isinstance(v, (int, float, str, bool))}
        }
        
        # Log as text
        hparam_str = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
        self.writer.add_text('Hyperparameters', hparam_str, 0)

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
        # Save final model
        model_path = self.save_dir / "agent_final.pth"
        self.agent.save(str(model_path))
        print(f"Saved final model to {model_path}")

        # Save training data
        np.save(self.save_dir / "episode_rewards.npy", self.episode_rewards)
        np.save(self.save_dir / "episode_lengths.npy", self.episode_lengths)

        # Save config used for this run
        config_path = self.save_dir / "config_used.yaml"
        self._save_config(config_path)

        # Plot rewards
        self._plot_rewards()

    def _save_config(self, path: Path):
        """Save the configuration used for this training run."""
        import yaml
        
        config_dict = {
            'experiment_name': self.config.experiment_name,
            'seed': self.config.seed,
            'env_name': self.config.env_name,
            'mode': self.config.mode,
            'opponent': self.config.opponent,
            'reward_shaping': self.config.reward_shaping,
            'agent_type': self.config.agent_type,
            'gamma': self.config.gamma,
            'tau': self.config.tau,
            'total_timesteps': self.config.total_timesteps,
            'learning_starts': self.config.learning_starts,
            'batch_size': self.config.batch_size,
            'replay_capacity': self.config.replay_capacity,
            'log_dir': self.config.log_dir,
            'save_interval': self.config.save_interval,
            'eval_interval': self.config.eval_interval,
            'eval_episodes': self.config.eval_episodes,
            **self.config.agent_params
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"Saved config to: {path}")

    def _plot_rewards(self):
        """Plot training rewards."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
            
            # Moving average
            window = min(50, len(self.episode_rewards) // 10)
            if len(self.episode_rewards) >= window and window > 1:
                moving_avg = np.convolve(self.episode_rewards, 
                                        np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(self.episode_rewards)), 
                        moving_avg, linewidth=2, label=f'{window}-Episode Moving Avg')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'{self.config.experiment_name}')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plot_path = self.save_dir / "training_rewards.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {plot_path}")
        except ImportError:
            print("Matplotlib not available, skipping plot")
