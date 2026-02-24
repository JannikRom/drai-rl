#!/bin/bash

#SBATCH --job-name=rl_paper
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

#SBATCH --partition=day

#SBATCH --gres=gpu:1

#SBATCH --time=10:00:00

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.polzer@student.uni-tuebingen.de

cd ~/drai-rl
CONFIGS=(
  "src/configs/lunar_lander/sac_lunarLander_er.yaml"
  "src/configs/lunar_lander/sac_lunarLander_per.yaml"
  "src/configs/lunar_lander/td3_lunarLander_pink_er.yaml"
  "src/configs/lunar_lander/td3_lunarLander_white_er.yaml"
  "src/configs/lunar_lander/td3_lunarLander_white_per.yaml"
)

SELECTED_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg python3 ~/drai-rl/src/train.py --config "$SELECTED_CONFIG"
