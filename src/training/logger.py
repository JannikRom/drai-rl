"""
Centralized TensorBoard + console logger driven by LoggingConfig.

Author: Jannik Rombach
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from common.logging_config import LoggingConfig


class TrainingLogger:

    def __init__(self, log_dir: Path, cfg: LoggingConfig) -> None:
        self.cfg    = cfg
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard: {log_dir}")
        print(f" tensorboard --logdir={log_dir}")

    # Losses
    def log_losses(self, losses: dict[str, Any], timestep: int) -> None:
        if not self.cfg.log_losses:
            return
        if timestep % self.cfg.loss_every_steps != 0:
            return

        self.writer.add_scalar("Loss/Critic", losses["critic_loss"], timestep)

        actor = losses.get("actor_loss")
        if actor is not None:
            self.writer.add_scalar("Loss/Actor", actor, timestep)

        if self.cfg.log_sac and "alpha" in losses:
            self.writer.add_scalar("SAC/Alpha", losses["alpha"], timestep)
            self.writer.add_scalar("SAC/Alpha_Loss", losses["alpha_loss"], timestep)
            self.writer.add_scalar("SAC/Mean_Log_Prob", losses["mean_log_prob"], timestep)

    def log_per_beta(self, beta: float, timestep: int) -> None:
        if not self.cfg.log_per:
            return
        if timestep % self.cfg.per_beta_every_steps != 0:
            return
        self.writer.add_scalar("PER/Beta", beta, timestep)

    # Episode
    def log_episode(
        self,
        episode_reward: float,
        episode_length: int,
        episode_num: int,
        timestep: int,
        extra: dict[str, float] | None = None,
    ) -> None:
        self.writer.add_scalar("Train/Episode_Reward", episode_reward, episode_num)
        self.writer.add_scalar("Train/Episode_Length", episode_length, episode_num)
        self.writer.add_scalar("Train/Episode_Reward_vs_Timestep", episode_reward, timestep)
        if extra:
            for key, value in extra.items():
                self.writer.add_scalar(key, value, episode_num)

    def log_moving_averages(
        self,
        rewards: list[float],
        lengths: list[float],
        episode_num: int,
    ) -> None:
        w = self.cfg.avg_window
        self.writer.add_scalar(
            f"Train/Avg_Reward_{w}ep", float(np.mean(rewards[-w:])), episode_num
        )
        self.writer.add_scalar(
            f"Train/Avg_Length_{w}ep", float(np.mean(lengths[-w:])), episode_num
        )

    def should_print(self, episode_num: int) -> bool:
        return episode_num % self.cfg.print_every_episodes == 0


    # Evaluation
    def log_eval(self, reward: float, timestep: int, tag: str = "Eval/Reward") -> None:
        if not self.cfg.log_eval:
            return
        self.writer.add_scalar(tag, reward, timestep)

    
    # Self-play pool
    def log_pool(
        self,
        pool_size: int,
        episode_num: int,
        win_rate: float | None = None,
        avg_reward: float | None = None,
    ) -> None:
        if not self.cfg.log_pool:
            return
        self.writer.add_scalar("SelfPlay/Pool_Size", pool_size, episode_num)
        if win_rate is not None:
            self.writer.add_scalar("SelfPlay/Pool_Update_WinRate", win_rate, episode_num)
        if avg_reward is not None:
            self.writer.add_scalar("SelfPlay/Pool_Update_AvgReward", avg_reward, episode_num)


    # Hyperparameters
    def log_hyperparameters(self, hparams: dict[str, Any]) -> None:
        text = "\n".join(f"{k}: {v}" for k, v in hparams.items())
        self.writer.add_text("Hyperparameters", text, 0)


    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
