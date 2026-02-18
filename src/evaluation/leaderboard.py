"""
TrueSkill-based leaderboard for agent evaluation.

Consumes MatchStats from MatchRunner, updates rating after each matchup,
and produces a ranked leaderboard sorted by LCB (lower confidence bound).

LCB = mu - 3 * sigma, where mu is the skill estimate and sigma is the uncertainty.

Author: Jannik Rombach
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import trueskill

from evaluation.match_runner import MatchStats


@dataclass
class AgentRating:
    """Trueskill rating + aggregated match history for one agent."""
    name: str
    mu: float = 25.0
    sigma: float = 8.333
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0
    total_games: int = 0

    @property
    def lcb(self) -> float:
        """Lower confidence bound — primary ranking metric."""
        return self.mu - 3.0 * self.sigma

    @property
    def win_rate(self) -> float:
        return self.total_wins / self.total_games if self.total_games > 0 else 0.0

    def summary_row(self, rank: int) -> str:
        return (
            f"{rank:<5} {self.name:<35} "
            f"{self.mu:>7.2f} {self.sigma:>7.2f} {self.lcb:>7.2f}  "
            f"W={self.total_wins} L={self.total_losses} D={self.total_draws} "
            f"({self.win_rate:.1%})"
        )
    
class Leaderboard:
    """Maintains TrueSkill rating for all agents and opponents."""

    HEADER = (
        f"{'Rank':<5} {'Agent':<35} "
        f"{'mu':>7} {'sigma':>7} {'LCB':>7}  Results"
    )

    def __init__(self):
        self._env = trueskill.TrueSkill(draw_probability=0.05)
        self._ratings: dict[str, trueskill.Rating] = {}
        self._records: dict[str, AgentRating] = {}


    def update(self, stats: MatchStats) -> None:
        """Update Trueskill ratings from a copletet MatchStats result."""
    
        agent_name = stats.agent_name
        opponent_name = stats.opponent_name

        self._ensure(agent_name)
        self._ensure(opponent_name)

        # Aggregate raw counts
        self._records[agent_name].total_wins += stats.wins
        self._records[agent_name].total_losses += stats.losses
        self._records[agent_name].total_draws += stats.draws
        self._records[agent_name].total_games += stats.n_episodes

        self._records[opponent_name].total_wins += stats.losses
        self._records[opponent_name].total_losses += stats.wins
        self._records[opponent_name].total_draws += stats.draws
        self._records[opponent_name].total_games += stats.n_episodes

        # Update Trueskill ratings
        if stats.wins > stats.losses:
            # Agent wins
            self._ratings[agent_name], self._ratings[opponent_name] = self._env.rate_1vs1(
                self._ratings[agent_name], self._ratings[opponent_name]
            )
        elif stats.wins < stats.losses:
            # Opponent wins
            self._ratings[opponent_name], self._ratings[agent_name] = self._env.rate_1vs1(
                self._ratings[opponent_name], self._ratings[agent_name]
            )
        
        # drawss no update

        self._records[agent_name].mu = self._ratings[agent_name].mu
        self._records[agent_name].sigma = self._ratings[agent_name].sigma
        self._records[opponent_name].mu = self._ratings[opponent_name].mu
        self._records[opponent_name].sigma = self._ratings[opponent_name].sigma

    def ranked(self) -> list[AgentRating]:
        """Return list of AgentRatings sorted by LCB."""
        return sorted(self._records.values(), key=lambda r: r.lcb, reverse=True)
    
    def best(self) -> AgentRating | None:
        """Return the current best agent (highest LCB)."""
        ranked = self.ranked()
        return ranked[0] if ranked else None
    
    def print(self) -> None:
        """Print the current leaderboard."""
        print("\n" + self.HEADER)
        print("-" * 75)
        for rank, record in enumerate(self.ranked(), start=1):
            print(record.summary_row(rank))
        print()

    def save(self, path: str) -> None:
        """Save full leaderboard state to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: asdict(record)
            for name, record in self._records.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Leaderboard saved to {path}")

    def load(self, path: str) -> None:
        """Restore leaderboard state from a previously saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        for name, record_data in data.items():
            self._records[name] = AgentRating(**record_data)
            self._ratings[name] = trueskill.Rating(
                mu=record_data["mu"], sigma=record_data["sigma"]
            )
        print(f"Leaderboard loaded from {path}")

    def _ensure(self, name: str) -> None:
        """Initialize rating entry if not yet seen."""
        if name not in self._ratings:
            self._ratings[name] = self._env.create_rating()
            self._records[name] = AgentRating(name=name)