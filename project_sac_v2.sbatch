#!/bin/bash

#SBATCH --job-name=hockey_rl
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G

#SBATCH --partition=week

#SBATCH --gres=gpu:1

#SBATCH --time=50:00:00

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriano.polzer@student.uni-tuebingen.de

cd ~/drai-rl-v2


singularity exec --nv /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg python3 ~/drai-rl-v2/src/train.py --config "src/configs/sac_job1.yaml"
