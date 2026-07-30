#!/bin/bash -l
#SBATCH --job-name=GPUJob
#SBATCH --output=GPUJob_results.%j.%N.txt
#SBATCH --error=GPUJob_errors.%j.%N.err
#SBATCH -p gpu
#SBATCH --gres=gpu:p100:2

module load miniconda/3
module load cuda
source activate recon
srun nvidia-smi
python -c 'import torch; print(torch.cuda.is_available())'
