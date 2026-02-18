"""
Universal training script for RL agents.

Usage:
    python train.py --config configs/sac/checkpoint1_pendulum.yaml
    python train.py --config configs/td3/checkpoint3_weak.yaml --seed 42
    
Author: Jannik Rombach
"""

import argparse
from common.config import RLConfig
from common.environments import get_env_dims
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent
from common.trainer import Trainer


def create_agent(config: RLConfig):
    """
    Create agent based on config.
    
    Args:
        config: RLConfig object
        
    Returns:
        Instantiated agent
        
    Raises:
        ValueError: If agent_type or env_name is unknown
    """
    # Get environment dimensions
    state_dim, action_dim, max_action = get_env_dims(config.env_name)
    
    # Create agent based on type
    agent_type = config.agent_type.lower()  # Make case-insensitive
    
    if agent_type == 'td3':
        return TD3Agent(state_dim, action_dim, max_action, config)
    
    elif agent_type == 'sac':
         return SACAgent(state_dim, action_dim, max_action, config)
    
    else:
        raise ValueError(
            f"Unknown agent type: '{config.agent_type}'. "
            f"Available: ['td3', 'sac', 'ppo']"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Train RL agent from config file'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override seed from config'
    )
    args = parser.parse_args()
    
    # Load config
    config = RLConfig.from_yaml(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        print(f"Overriding seed: {config.seed} → {args.seed}")
        config.seed = args.seed
    
    # Create agent
    agent = create_agent(config)
    
    # Create trainer and train
    trainer = Trainer(agent, config)
    trainer.train()
    
    print("\n== Training complete ==")
    print(f"Results saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()
