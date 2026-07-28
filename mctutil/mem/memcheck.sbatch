#!/bin/bash -l
#SBATCH --job-name=MemCheck

# Configure the cluster environment before submission so mctutil is on PATH.
mctutil mem clean --dry-run
