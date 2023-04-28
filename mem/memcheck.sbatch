#!/bin/bash -l
#SBATCH --job-name=MemClean
#SBATCH --output=../log/memcheck_%N.std
#SBATCH --error=../log/memcheck_%N.err
#SBATCH -p compute

module load miniconda/3
source activate recon

python clean_shared.py False
