#!/bin/bash -l
#SBATCH --job-name=MemClean

# Configure the cluster environment before submission so mctutil is on PATH.
mctutil mem clean --execute
