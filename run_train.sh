#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=res_train.txt
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH -w hlt01

echo "activate env..."
source activate pyt-gpu   

echo "running code..."
python -u train_model.py
