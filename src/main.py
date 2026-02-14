import gymnasium as gym
from common.config import RLConfig
from agents.td3_agent import TD3Agent
from trainer import Trainer


def main():
    """Train TD3 agent on Pendulum-v1."""
    # Load configuration
    config = RLConfig.from_yaml("configs/td3_pendulum.yaml")
    
    # Get environment dimensions
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()
    
    # Create agent
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        config=config.agent_params
    )
    
    # Train
    trainer = Trainer(agent, config)
    trainer.train()


if __name__ == "__main__":
    main()
