from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np
from comprl.client import Agent, launch_client

from common.config import RLConfig
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent
from train import create_agent

# === Competition Agent Configs ===
AGENTS = {
    "drai_sac": {
        "config": "./final_model/drai_sac/sac_final.yaml",
        "checkpoint": "./final_model/drai_sac/sac_final.pth",
    },
    "drai_td3": {
        "config": "./final_model/drai_td3/td3_final.yaml",
        "checkpoint": "./final_model/drai_td3/td3_final.pth",
    },
    "drai_team": {
        "config": "./final_model/drai_team/team_final.yaml",
        "checkpoint": "./final_model/drai_team/team_final.pth",
    },
}
    

class CompetitionAgent(Agent):
    def __init__(self, agent_key: str):
        super().__init__()
        cfg = AGENTS[agent_key]
        self.config = RLConfig.from_yaml(cfg["config"])
        self.agent = create_agent(self.config)
        print(f"Load checkpoint: {cfg['checkpoint']}")
        self.agent.load(cfg["checkpoint"])

    def get_step(self, observation: list[float]) -> list[float]:
        obs_array = np.array(observation)
        action = self.agent.select_action(obs_array, eval_mode=True)
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["drai_sac", "drai_td3", "drai_team", "weak", "strong", "random"],
        default="weak",
        help="Which agent to use.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent in AGENTS:
        agent = CompetitionAgent(args.agent)
    elif args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()