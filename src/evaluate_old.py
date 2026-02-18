"""
Universal evaluation script for trained RL agents.

Usage:
    python evaluate.py --checkpoint logs/td3/experiment/agent_step_100000.pth --config configs/td3/checkpoint1_lunarlander.yaml --episodes 10
    python evaluate.py --checkpoint logs/sac/experiment/agent_step_50000.pth --config configs/sac/checkpoint1_pendulum.yaml --episodes 5 --render --render-every 5

Author: Jannik Rombach
"""

import argparse
import numpy as np
from pathlib import Path
from common.config import RLConfig
from common.environments import get_env_dims, make_env
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent


def create_agent(config: RLConfig):
    """
    Create agent based on config.
    """
    state_dim, action_dim, max_action = get_env_dims(config.env_name)
    
    agent_type = config.agent_type.lower()
    
    if agent_type == 'td3':
        return TD3Agent(state_dim, action_dim, max_action, config)
    elif agent_type == 'sac':
        return SACAgent(state_dim, action_dim, max_action, config)
    else:
        raise ValueError(
            f"Unknown agent type: '{config.agent_type}'. "
            f"Available: ['td3', 'sac']"
        )


def evaluate_agent(agent, env, num_episodes: int = 10, render: bool = False, render_every: int = 1):
    """
    Evaluate agent over multiple episodes without exploration noise.
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        step = 0
        
        while not done:
            if render and step % render_every == 0:
                env.render()
            
            # No exploration noise during evaluation
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Length: {episode_length}")
    
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained RL agent'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.pth)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes (default: 10)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during evaluation'
    )
    parser.add_argument(
    '--render-every',
    type=int,
    default=1,
    help='Render every Nth frame (default: 1, higher = faster)'
    )

    args = parser.parse_args()
    
    # Load config
    config = RLConfig.from_yaml(args.config)
    print(f"Loaded config: {config.experiment_name}")
    print(f"Agent type: {config.agent_type}")
    print(f"Environment: {config.env_name}")
    
    # Create environment
    env = make_env(config.env_name, config)
    env.reset(seed=config.seed)
    
    # Create agent
    agent = create_agent(config)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    agent.load(str(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path.name}")
    
    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes...")
    episode_rewards, episode_lengths = evaluate_agent(
        agent, env, num_episodes=args.episodes, render=args.render, render_every=args.render_every
    )
    
    # Statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint:    {checkpoint_path.name}")
    print(f"Episodes:      {args.episodes}")
    print(f"Mean Reward:   {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length:   {mean_length:.2f}")
    print(f"Min Reward:    {min_reward:.2f}")
    print(f"Max Reward:    {max_reward:.2f}")
    print("="*60)
    
    env.close()


if __name__ == '__main__':
    main()
