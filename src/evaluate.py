"""
Universal evaluation entry point for trained RL agents.

Two modes, selected automatically from config:

  Standard mode  — any Gymnasium environment (Pendulum, LunarLander, etc.)
    python evaluate.py --agent logs/td3/.../agent_final.pth
                       --config configs/td3/checkpoint1_lunarlander.yaml
                       --episodes 20 --render

  Hockey tournament mode — full round-robin with TrueSkill leaderboard
    python evaluate.py --agent logs/td3/.../agent_final.pth logs/td3/.../agent_step_500000.pth
                       --config configs/td3/checkpoint3_hockey.yaml
                                configs/td3/checkpoint3_hockey_pink.yaml
                       --episodes 200 --load-leaderboard logs/eval/leaderboard.json

Author: Jannik Rombach
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from PIL import Image

from common.config import RLConfig
from environments.environments import get_env_dims, make_env
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent
import hockey.hockey_env as hockey_env

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
    return agent, config


def run_standard_eval(agent, config: RLConfig, n_episodes: int, render: bool, render_every: int = 1, record: bool = False, record_path: str = "agent.gif"):
    """Evaluate agent on any standard Gymnasium environment."""
    render_mode = "rgb_array" if record else ("human" if render else None)
    env = make_env(config.env_name, config, render_mode=render_mode)

    if render and not record:
        import pygame
        pygame.init()

    episode_rewards = []
    episode_lengths = []
    frames = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)

            if record:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))
            elif render and step % render_every == 0:
                import pygame
                pygame.event.pump()

            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            step += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"  Episode {ep + 1:3d}/{n_episodes} | "
              f"Reward: {episode_reward:8.2f} | Length: {episode_length}")

    env.close()

    if record and frames:
        frames[0].save(
            record_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=33,  # ~30fps
        )
        print(f"\nGIF saved to: {record_path}")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Environment:  {config.env_name}")
    print(f"Checkpoint:   {config.experiment_name}")
    print(f"Episodes:     {n_episodes}")
    print(f"Mean Reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min / Max:    {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"Mean Length:  {np.mean(episode_lengths):.1f}")
    print("=" * 60)

# Hockey specific
def run_hockey_eval(agents: dict, n_episodes: int, render: bool, save_dir: str, tag: str, load_leaderboard: str):
    """Full round-robin tournament with TrueSkill"""
    import hockey.hockey_env as hockey_env
    from evaluation.leaderboard import Leaderboard
    from evaluation.match_runner import MatchRunner

    def make_hockey_env():
        return hockey_env.HockeyEnv(mode='NORMAL')
    
    opponents = {
        "basic_weak":   hockey_env.BasicOpponent(weak=True),
        "basic_strong": hockey_env.BasicOpponent(weak=False),
    }
    all_participants = {**opponents, **agents}

    leaderboard = Leaderboard()
    if load_leaderboard:
        leaderboard.load(load_leaderboard)

    runner = MatchRunner(env_fn=make_hockey_env, render=render)

    n = len(all_participants)
    total_matchups = n * (n - 1)
    print(f"\nTournament: {n} participants, {total_matchups} matchups, "
          f"{n_episodes} episodes each\n")
    
    names = list(all_participants.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    completed = 0

    for name_a, name_b in pairs:
        for agent_name, opp_name in [(name_a, name_b), (name_b, name_a)]:
            completed += 1
            total = len(pairs) * 2
            print(f"[{completed}/{total}] {agent_name} vs {opp_name} ...")
            stats = runner.run(
                agent=all_participants[agent_name],
                agent_name=agent_name,
                opponent=all_participants[opp_name],
                opponent_name=opp_name,
                n_episodes=n_episodes,
                seed=completed * 100,
            )
            leaderboard.update(stats)
            print(f"       {stats.summary()}")

    leaderboard.print()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(save_dir) / f"{tag}_leaderboard.json")
    leaderboard.save(save_path)
    print(f"\nLeaderboard saved to: {save_path}")

def record_single_match(agent, agent_name: str, n_episodes: int, record_path: str):
    """Record a single match of the agent vs the strong opponent as a GIF."""
    
    env = hockey_env.HockeyEnv(mode='NORMAL')
    opponent = hockey_env.BasicOpponent(weak=False)
    frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        obs_opponent = env.obs_agent_two()
        done = False

        while not done:
            if hasattr(agent, 'select_action'):
                action_agent = agent.select_action(obs, eval_mode=True)
            else:
                action_agent = agent.act(obs)
            action_opponent = opponent.act(obs_opponent)

            combined = np.hstack([action_agent, action_opponent])
            obs, _, terminated, truncated, _ = env.step(combined)
            obs_opponent = env.obs_agent_two()

            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(Image.fromarray(frame))

            done = terminated or truncated

    env.close()

    if frames:
        frames[0].save(
            record_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration= 32,
        )
        print(f"GIF saved to: {record_path}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")

    parser.add_argument("--agent",  nargs="+", required=True, metavar="CHECKPOINT")
    parser.add_argument("--config", nargs="+", required=True, metavar="CONFIG")
    parser.add_argument("--name",   nargs="+", default=None,  metavar="NAME")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="logs/eval")
    parser.add_argument("--load-leaderboard", type=str, default=None, metavar="JSON")
    parser.add_argument("--tag", type=str, default="eval")
    parser.add_argument("--record-match", action="store_true")
    parser.add_argument("--record-path", type=str, default="agent.gif")
    parser.add_argument("--record-episodes", type=int, default=1)

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
    first_config = None
    for i, (checkpoint, config_path) in enumerate(zip(args.agent, args.config)):
        agent, config = load_agent(checkpoint, config_path)
        name = args.name[i] if args.name else (config.experiment_name or Path(checkpoint).stem)
        agents[name] = agent
        if first_config is None:
            first_config = config
        print(f"Loaded: {name} from {checkpoint}")

    # Auto-select mode based on env_name
    is_hockey = "hockey" in first_config.env_name.lower()

    if is_hockey:
        if len(agents) == 0:
            print("ERROR: Hockey mode requires at least one agent.")
            sys.exit(1)
        name, agent = next(iter(agents.items()))
        if args.record_match:
            record_single_match(
                agent=agent,
                agent_name=name,
                n_episodes=args.record_episodes,
                record_path=args.record_path,
            )
        run_hockey_eval(
            agents=agents,
            n_episodes=args.episodes,
            render=args.render,
            save_dir=args.save_dir,
            tag=args.tag,
            load_leaderboard=args.load_leaderboard,
        )
    else:
        if len(agents) > 1:
            print("WARNING: Standard mode only evaluates the first agent. "
                  "Use hockey mode for multi-agent tournaments.")
        name, agent = next(iter(agents.items()))
        run_standard_eval(
            agent=agent,
            config=first_config,
            n_episodes=args.episodes,
            render=args.render,
            render_every=args.render_every
        )


if __name__ == "__main__":
    main()

