#!/bin/sh

#SBATCH --job-name=DownsampleShapenet
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out

python superdec/data/downsample_shapenet.py