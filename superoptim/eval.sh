#!/bin/sh

#SBATCH --job-name=SuperOptim
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out

# Default type if not provided
TYPE=${1:-empty}

# python -m superoptim.batch_evaluate --type "$TYPE"
python -m superoptim.batch_evaluate --type "$TYPE" --prefix abo_train