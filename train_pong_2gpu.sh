#!/bin/bash -login

#SBATCH --job-name=train_pong
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=3-12:00:00

module load CUDA
module load apps/ffmpeg/4.3

echo "Training on Pong..."
python train_pong.py
