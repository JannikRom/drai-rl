"""Logging configuration parsed from the YAML config file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LoggingConfig:
    # TensorBoard scalar intervals (in timesteps)
    loss_every_steps: int
    per_beta_every_steps: int

    # Console print interval (in episodes)
    print_every_episodes: int

    # Moving-average window for console + TensorBoard
    avg_window: int

    # Which scalar groups to enable
    log_losses: bool
    log_per: bool
    log_sac: bool
    log_eval: bool
    log_pool: bool  # self-play only

    # Final artefacts
    save_reward_plot: bool
    save_numpy_arrays: bool
    save_config_yaml: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "LoggingConfig":
        """
        Build from the 'logging' sub-dict in a YAML config.

        Raises:
            KeyError:   A required logging key is missing from the config.
            TypeError:  The 'logging' block is present but not a dict.
        """
        if d is None:
            raise KeyError(
                "'logging' block is missing from config. "
                "All logging fields must be set explicitly."
            )
        if not isinstance(d, dict):
            raise TypeError(
                f"'logging' must be a YAML mapping, got {type(d).__name__}."
            )

        required = set(cls.__dataclass_fields__)
        missing  = required - d.keys()
        if missing:
            raise KeyError(
                f"Missing required logging keys: {sorted(missing)}"
            )

        unknown = d.keys() - required
        if unknown:
            raise KeyError(
                f"Unknown logging keys: {sorted(unknown)}"
            )

        return cls(**d)
