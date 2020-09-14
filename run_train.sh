#!/bin/bash
#SBATCH --job-name=train_1
#SBATCH --output=res_4_1_1.txt
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH -w hlt01

echo "activate env..."
source activate pyt-gpu   

echo "running code..."
python -u train_model.py
