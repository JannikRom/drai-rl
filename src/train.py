"""
Universal training entry point.

Usage:
    # Standard training
    python train.py --config configs/td3/hockey_strong.yaml

    # Override seed
    python train.py --config configs/td3/hockey_strong.yaml --seed 42

    # Self-play from scratch
    python train.py --config configs/td3/selfplay_v1.yaml

    # Self-play from checkpoint
    python train.py --config configs/td3/selfplay_v1.yaml --checkpoint logs/td3/run1/agent_final.pth

Author: Jannik Rombach
"""

from __future__ import annotations

import argparse

from common.config import RLConfig
from environments.environments import get_env_dims
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent
from training.standard_trainer import StandardTrainer
from training.selfplay_trainer import SelfPlayTrainer

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

def create_trainer(agent: TD3Agent | SACAgent, config: RLConfig) -> StandardTrainer | SelfPlayTrainer:
    """
    Create trainer based on config.
    """
    training_mode = config.get("training_mode").lower()
    
    if training_mode == 'selfplay':
        return SelfPlayTrainer(agent, config)
    
    elif training_mode == 'standard':
        return StandardTrainer(agent, config)
    
    else:
        raise ValueError(
            f"Unknown training mode: '{training_mode}'. "
            f"Available: ['standard', 'selfplay']"
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

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        metavar='PATH',
        help='Load pretrained agent from checkpoint before training',
    )

    args = parser.parse_args()
    
    # Load config
    config = RLConfig.from_yaml(args.config)
    
    if args.seed is not None:
        print(f"Overriding seed: {config.seed} → {args.seed}")
        config.seed = args.seed
    
    # Create agent
    agent = create_agent(config)

    if args.checkpoint is not None:
        print(f"Loading agent from checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    
    trainer = create_trainer(agent, config)
    trainer.train()
    
    print(f"\n== Training complete — results in: {trainer.save_dir} ==")


if __name__ == "__main__":
    main()
