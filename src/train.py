"""
Universal training script for RL agents.

Usage:
    # Standard training (config: training_mode: "standard")
    python train.py --config configs/td3/checkpoint3_hockey_strong_pink.yaml

    # Override seed
    python train.py --config configs/td3/checkpoint3_hockey_strong_pink.yaml --seed 42

    # Self-play training from scratch (config: training_mode: "selfplay")
    python train.py --config configs/td3/selfplay_v1.yaml

    # Self-play fine-tune from existing checkpoint
    python train.py --config configs/td3/selfplay_v1.yaml \
                    --checkpoint logs/td3/checkpoint3_hockey_strong_td3_pink/agent_final.pth

Author: Jannik Rombach
"""

import argparse

from common.config import RLConfig
from environments.environments import get_env_dims
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent
from training.trainer import Trainer
from training.selfplay_trainer import SelfPlayTrainer

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
    
    training_mode = config.get("training_mode")
    if training_mode == 'selfplay':
        print("Training mode: self-play")
        trainer = SelfPlayTrainer(agent, config)
    else:
        print("Training mode: standard")
        trainer = Trainer(agent, config)

    trainer.train()
    
    print("\n== Training complete ==")
    print(f"Results saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()
