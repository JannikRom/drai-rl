import argparse
from pathlib import Path
from common.config import RLConfig
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent
from trainer import Trainer


def create_agent(config: RLConfig):
    """Create agent based on config."""
    
    if config.agent_type == "TD3":
        return TD3Agent(
            state_dim=config.get('state_dim'),
            action_dim=config.get('action_dim'),
            max_action=config.get('max_action', 1.0),
            config=config.agent_params
        )
    
    elif config.agent_type == "SAC":
       return SACAgent(
            state_dim=config.get('state_dim'),
            action_dim=config.get('action_dim'),
            max_action=config.get('max_action', 1.0),
            config=config.agent_params
       )
    #elif config.agent_type == "DQN":
    #    return DQNAgent(...)
    
    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    config = RLConfig.from_yaml(args.config)
    
    # Create agent
    agent = create_agent(config)
    
    # Create trainer and train
    trainer = Trainer(agent, config)
    trainer.train()


if __name__ == "__main__":
    main()
