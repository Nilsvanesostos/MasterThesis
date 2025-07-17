#!/bin/bash
#SBATCH --job-name=SMGM_uncond
#SBATCH --output=checkpoints.out 
#SBATCH --error=main.err
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=10G
#SBATCH --gres=gpu:1 
#SBATCH --time=15:00:00
#SBATCH --partition=allgroups

python SMGM_uncond.py train --epochs 500 --checkpoint_interval 80 --batch_size 64 --learning_rate 2e-3 --dataset "mnist" --checkpoint_dir "./checkpoints/unconditional/SM/mnist/" --schedule "linear"

