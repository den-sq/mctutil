#!/bin/bash -l
#SBATCH --job-name=remote_work
#SBATCH --output=../log/ImageJ_%j_%N.std
#SBATCH --error=../log/ImageJ_%j_%N.err
#SBATCH -p dense
#SBATCH --exclusive
#SBATCH --cpus-per-task=2
#SBATCH --mem=312G

../Fiji.app/ImageJ-linux64
