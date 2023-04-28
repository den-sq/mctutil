#!/bin/bash -l
#SBATCH --job-name=Globus
#SBATCH --output=../log/globus_%j_%N.std
#SBATCH --error=../log/globus_%j_%N.err
#SBATCH -p compute
#SBATCH --extra-node-info=48:1:1

module load miniconda/3
../globusconnectpersonal/globusconnectpersonal -start
