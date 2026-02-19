"""
Opponent pool for self-play training.

Maintains a pool of past agent snapshots sampled ranodmly during training.
Random pool sampling prevents overfitting.

Author: Jannik Rombach
"""

from __future__ import annotations

import copy
import random

class OpponentPool:
    """
    Pool of past agent snapshots for self-play training.
    """

    def __init__(self, max_size: int = 20, initial_opponent=None):

        self.max_size = max_size
        self._pool: list = []

        if initial_opponent is not None:
            self._pool.append(copy.deepcopy(initial_opponent))
    
    def add(self, agent, name: str = None) -> None:
        """
        Copy the current agent and add it to the pool.
        Dropy the oldest entry if max_size is exceeded.
        """

        snapshot = copy.deepcopy(agent)

        if hasattr(snapshot, "actor"):
            snapshot.actor.eval()
        if hasattr(snapshot, "critic"):
            snapshot.critic.eval()

        snapshot._pool_name = name or agent.__class__.__name__

        self._pool.append(snapshot)

        if len(self._pool) > self.max_size:
            self._pool.pop(0)
    
    def sample(self) -> object:
        """
        Uniformly sample a random opponent from the pool.
        """
        if not self._pool:
            return None
        return random.choice(self._pool)
    
    def members(self) -> list:
        return list(self._pool)
    
    def __len__(self) -> int:
        return len(self._pool)
    
    def __repr__(self):
        return f"OpponentPool(size={len(self._pool)}, max_size={self.max_size})"
