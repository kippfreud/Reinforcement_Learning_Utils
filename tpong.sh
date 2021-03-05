#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3-12:00:00
#SBATCH --partition gpu
#SBATCH --mem=16G
#SBATCH --job-name=train_pong
module load CUDA
module load apps/torch/28.01.2019
module load languages/anaconda3/3.6.5
echo "Training on Pong..."
python3 train_pong.py > train_pong.out