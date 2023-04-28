#!/bin/bash -l
#SBATCH --job-name=code_tunnel
#SBATCH --output=../log/code_%j_%N.std
#SBATCH --error=../log/code_%j_%N.err
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --extra-node-info=48:1:1
#SBATCH --time=480

module load git
module load miniconda/3

../code_tunnel/code tunnel
