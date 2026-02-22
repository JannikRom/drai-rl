"""
RL utilities: agent creation, device setup, etc.
"""

from environments.environments import get_env_dims
from common.config import RLConfig
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
