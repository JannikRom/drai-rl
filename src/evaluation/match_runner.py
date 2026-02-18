"""
Hockey match runner for agent evaluation.

Runs N episodes between an agent and an opponent, collecting
win/loss/draw statistics and per-episode rewards.

Author: Jannik Rombach
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MatchStats:
    """Structured result of a multiple-episode match."""
    agent_name: str
    opponent_name: str
    n_episodes: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    episode_rewards: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_episodes if self.n_episodes > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.n_episodes if self.n_episodes > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.n_episodes if self.n_episodes > 0 else 0.0

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0

    def summary(self) -> str:
        return (
            f"{self.agent_name} vs {self.opponent_name} | "
            f"W={self.wins} L={self.losses} D={self.draws} "
            f"({self.win_rate:.1%} WR) | "
            f"avg_r={self.avg_reward:.2f} ± {self.std_reward:.2f}"
        )


class MatchRunner:
    """
    Runs evaluation matches between any two participants inside the HockeyEnv.
    Supports BasicOpponent (.act) and trained agents (.select_action) on both sides.
    """

    def __init__(self, env_fn):
        self.env_fn = env_fn

    def _get_action(self, participant, obs: np.ndarray) -> np.ndarray:
        """Unified action interface for BasicOpponent and trained agents."""
        if hasattr(participant, 'select_action'):
            return participant.select_action(obs, eval_mode=True)
        elif hasattr(participant, 'act'):
            return participant.act(obs)
        else:
            raise TypeError(f"Unsupported participant type: {type(participant)}")

    def run(
            self,
            agent,
            agent_name: str,
            opponent,
            opponent_name: str,
            n_episodes: int = 100,
            seed: int = 0,
    ) -> MatchStats:
        """Run n_episodes of a match between agent and opponent."""

        env = self.env_fn()
        stats = MatchStats(
            agent_name=agent_name,
            opponent_name=opponent_name,
            n_episodes=n_episodes,
        )

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            obs_agent = obs
            obs_opponent = env.obs_agent_two()

            if hasattr(agent, 'reset_exploration'):
                agent.reset_exploration()

            done = False
            ep_reward = 0.0

            while not done:
                action_agent = self._get_action(agent, obs_agent)
                action_opponent = self._get_action(opponent, obs_opponent)

                combined = np.hstack([action_agent, action_opponent])
                obs, reward, terminated, truncated, info = env.step(combined)
                done = terminated or truncated

                ep_reward += reward
                obs_agent = obs
                obs_opponent = env.obs_agent_two()

            stats.episode_rewards.append(ep_reward)
            winner = info.get('winner', 0)
            if winner == 1:
                stats.wins += 1
            elif winner == -1:
                stats.losses += 1
            else:
                stats.draws += 1

        env.close()
        return stats
