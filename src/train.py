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
from agents.create_agent import create_agent
from agents.base_agent import BaseAgent
from training.standard_trainer import StandardTrainer
from training.selfplay_trainer import SelfPlayTrainer
from training.report_trainer import ReportTrainer


def create_trainer(agent: BaseAgent, config: RLConfig) -> StandardTrainer | SelfPlayTrainer:
    """
    Create trainer based on config.
    """
    training_mode = config.get("training_mode").lower()
    
    if training_mode == "report":
        return ReportTrainer(agent, config)
    elif training_mode == 'selfplay':
        return SelfPlayTrainer(agent, config)
    
    elif training_mode == 'standard':
        return StandardTrainer(agent, config)
    
    else:
        raise ValueError(
            f"Unknown training mode: '{training_mode}'. "
            f"Available: ['standard', 'selfplay', 'strong_weak']"
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
