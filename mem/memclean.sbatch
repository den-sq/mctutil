#!/bin/bash -l
#SBATCH --job-name=MemClean
#SBATCH --output=../log/memclean_%N.std
#SBATCH --error=../log/memclean_%N.err
#SBATCH -p compute

module load miniconda/3
source activate recon

python clean_shared.py True
