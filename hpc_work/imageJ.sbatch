#!/bin/bash -l
#SBATCH --job-name=remote_work
#SBATCH --output=../log/ImageJ_%j_%N.std
#SBATCH --error=../log/ImageJ_%j_%N.err
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --extra-node-info=48:1:1

../Fiji.app/ImageJ-linux64 &
