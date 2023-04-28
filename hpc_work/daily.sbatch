#!/bin/bash -l
#SBATCH --job-name=remote_work
#SBATCH --output=../log/daily_%j_%N.std
#SBATCH --error=../log/daily_%j_%N.err
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --extra-node-info=48:1:1
#SBATCH --time=480

module load git
module load miniconda/3

../Fiji.app/ImageJ-linux64 &
../code_tunnel/code tunnel
