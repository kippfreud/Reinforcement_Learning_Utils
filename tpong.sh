#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3-12:00:00
#SBATCH --partition gpu
#SBATCH --mem=16G
#SBATCH --job-name=train_pong
echo "Training on Pong..."
python train_pong.py
