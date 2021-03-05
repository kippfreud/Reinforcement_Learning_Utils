#!/bin/bash -login

#SBATCH --job-name=train_pong
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-12:00:00

module load CUDA

echo "Training on Pong..."
python train_pong.py
