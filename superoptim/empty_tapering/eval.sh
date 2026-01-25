#!/bin/sh

#SBATCH --job-name=SuperOptim
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=24:00:00

python -m superoptim.empty_tapering.batch_evaluate