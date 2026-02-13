#!/bin/sh

#SBATCH --job-name=SuperDecABO
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=a100_80:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train-%j.out

python train/train.py --config-name=train_abo
python train/train.py --config-name=train_abo run_name=abo_iou loss.type=iou