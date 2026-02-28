# Training

This directory contains the training infrastructure for both SAC and TD3 agents.

## Trainers

- **standard_trainer.py** — Base trainer for training against fixed opponents or the gymnasium simple environments.
- **selfplay_trainer.py** — Extends standard training with a self-play loop, periodically adding agent snapshots to the opponent pool.

## Other Files

- **opponent_pool.py** — Manages the opponent pool, including snapshot storage and recency-biased sampling.
- **logger.py** — Handles TensorBoard logging of training metrics.
- **strong_sac.pth / strong_td3.pth** — Pre-trained model weights used as strong baseline opponents.
- **strong_sac.yaml / strong_td3.yaml** — Corresponding configuration files for the pre-trained models.

- **/fixed_opponents** - Pre-trained models for more diverse training.
