#!/bin/bash
#SBATCH --job-name=drai_hockey
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jannik.rombach@student.uni-tuebingen.de

set -euo pipefail

REPO=/home/stud342/drai-rl
cd "$REPO"

CONFIGS=(
  "$REPO/src/configs/td3/jobs/sac_job1.yaml"
  "$REPO/src/configs/td3/jobs/td3_job2.yaml"
  "$REPO/src/configs/td3/jobs/td3_job3.yaml"
)

HOST_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "HOST_CONFIG=$HOST_CONFIG"
ls -lah "$HOST_CONFIG"

CONTAINER_CONFIG="${HOST_CONFIG/$REPO/\/host_src}"

CHECKPOINT="/host_src/src/logs/td3_final_logs/td3_3M/td3_3M.pth"

singularity exec --nv -B "$REPO:/host_src" \
  /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg \
  python3 /host_src/src/train.py \
    --config "$CONTAINER_CONFIG" \
    --checkpoint "$CHECKPOINT"