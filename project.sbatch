#!/bin/bash

#SBATCH --job-name=hockey_rl
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G

#SBATCH --partition=week

#SBATCH --gres=gpu:1

#SBATCH --time=40:00:00

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.polzer@student.uni-tuebingen.de

cd ~/drai-rl
CONFIGS=(
  "src/configs/cluster/checkpoint3_strong_baseline_er.yaml"
  "src/configs/cluster/checkpoint3_strong_baseline_per.yaml"
  "src/configs/cluster/checkpoint4_selfplay_baseline_er.yaml"
  "src/configs/cluster/checkpoint4_selfplay_custom_rewards.yaml"
  "src/configs/cluster/chekpoint3_strong_custom_reward_high_er.yaml"
  "src/configs/cluster/chekpoint3_strong_custom_reward_low_er.yaml"
)

SELECTED_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg python3 ~/drai-rl/src/train.py --config "$SELECTED_CONFIG"
