#!/bin/sh

#SBATCH --job-name=PrepareABO
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out

python superdec/data/prepare_abo.py