"""
Evaluation entry point for trained RL agents.

Loads one or more agent checkpoints, runs them against a fixed
opponent suite, builds a TrueSkill leaderboard, and saves results.

Usage:
    # Evaluate a single checkpoint
    python evaluate.py --agent logs/td3/checkpoint3_hockey_strong_td3_pink/agent_final.pth
                       --config configs/td3/checkpoint3_hockey_strong_pink.yaml
                       --episodes 200

    # Evaluate multiple checkpoints and rank them
    python evaluate.py --agent logs/td3/.../agent_final.pth logs/td3/.../agent_step_500000.pth
                       --config configs/td3/checkpoint3_hockey_strong_pink.yaml
                                configs/td3/checkpoint3_hockey_strong.yaml
                       --episodes 200

    # Load a previous leaderboard and add new agents to it
    python evaluate.py --agent logs/td3/.../agent_final.pth
                       --config configs/td3/checkpoint3_hockey_strong_pink.yaml
                       --load-leaderboard logs/eval/leaderboard.json

Author: Jannik Rombach
"""

import argparse
import sys
from pathlib import Path

import hockey.hockey_env as hockey_env

from common.config import RLConfig
from common.environments import get_env_dims
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent
from evaluation.match_runner import MatchRunner
from evaluation.leaderboard import Leaderboard


def make_hockey_env():
    return hockey_env.HockeyEnv(mode='NORMAL')


def load_agent(checkpoint_path: str, config_path: str):
    """Load a trained agent from a checkpoint file using its config."""
    config = RLConfig.from_yaml(config_path)
    state_dim, action_dim, max_action = get_env_dims(config.env_name)

    agent_type = config.agent_type.lower()
    if agent_type == 'td3':
        agent = TD3Agent(state_dim, action_dim, max_action, config)
    elif agent_type == 'sac':
        agent = SACAgent(state_dim, action_dim, max_action, config)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    agent.load(checkpoint_path)
    return agent, config.experiment_name


def build_opponent_suite() -> dict:
    """Returns the fixed set of basic opponents included in every tournament."""
    return {
        "basic_weak": hockey_env.BasicOpponent(weak=True),
        "basic_strong": hockey_env.BasicOpponent(weak=False),
    }


def run_tournament(
        all_participants: dict,
        leaderboard: Leaderboard,
        runner: MatchRunner,
        n_episodes: int
    ) -> None:
    """
    Full round-robin tournament: every participant plays as agent against
    every other participant from both sides (player 1 and player 2).
    """
    names = list(all_participants.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    total = len(pairs) * 2
    completed = 0

    for name_a, name_b in pairs:
        participant_a = all_participants[name_a]
        participant_b = all_participants[name_b]

        completed += 1
        print(f"[{completed}/{total}] {name_a} vs {name_b} ({n_episodes} episodes) ...")
        stats = runner.run(
            agent=participant_a,
            agent_name=name_a,
            opponent=participant_b,
            opponent_name=name_b,
            n_episodes=n_episodes,
            seed=completed * 100,
        )
        leaderboard.update(stats)
        print(f"       {stats.summary()}")

        completed += 1
        print(f"[{completed}/{total}] {name_b} vs {name_a} ({n_episodes} episodes) ...")
        stats = runner.run(
            agent=participant_b,
            agent_name=name_b,
            opponent=participant_a,
            opponent_name=name_a,
            n_episodes=n_episodes,
            seed=completed * 100,
        )
        leaderboard.update(stats)
        print(f"       {stats.summary()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")

    parser.add_argument(
        "--agent",
        nargs="+",
        required=True,
        metavar="CHECKPOINT",
        help="Path(s) to agent .pth checkpoint files",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        metavar="CONFIG",
        help="Path(s) to YAML config files — must match --agent order",
    )
    parser.add_argument(
        "--name",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Optional display names for agents — must match --agent order",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of episodes per matchup (default: 200)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="logs/eval",
        help="Directory to save leaderboard JSON (default: logs/eval)",
    )
    parser.add_argument(
        "--load-leaderboard",
        type=str,
        default=None,
        metavar="JSON",
        help="Path to a previously saved leaderboard JSON to continue from",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="eval",
        help="Tag appended to saved leaderboard filename (default: eval)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.agent) != len(args.config):
        print("ERROR: --agent and --config must have the same number of entries.")
        sys.exit(1)

    if args.name and len(args.name) != len(args.agent):
        print("ERROR: --name must have the same number of entries as --agent.")
        sys.exit(1)

    agents = {}
    for i, (checkpoint, config_path) in enumerate(zip(args.agent, args.config)):
        agent, experiment_name = load_agent(checkpoint, config_path)
        name = args.name[i] if args.name else (experiment_name or Path(checkpoint).stem)
        agents[name] = agent
        print(f"Loaded: {name} from {checkpoint}")

    all_participants = build_opponent_suite()
    all_participants.update(agents)

    leaderboard = Leaderboard()
    if args.load_leaderboard:
        leaderboard.load(args.load_leaderboard)

    runner = MatchRunner(env_fn=make_hockey_env)

    n = len(all_participants)
    total_matchups = n * (n - 1)
    print(f"\nTournament: {n} participants, {total_matchups} matchups, "
          f"{args.episodes} episodes each\n")

    run_tournament(
        all_participants=all_participants,
        leaderboard=leaderboard,
        runner=runner,
        n_episodes=args.episodes,
    )

    leaderboard.print()

    save_path = str(Path(args.save_dir) / f"{args.tag}_leaderboard.json")
    leaderboard.save(save_path)


if __name__ == "__main__":
    main()
