#!/bin/bash -l
#SBATCH --job-name=AAA595_Mosaic_Center_Section9_Center
#SBATCH --output=job/res.%j.%N.txt
#SBATCH --error=job/err.%j.%N.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --partition=dense
#SBATCH --mail-user=den5123@psu.edu
#SBATCH --exclude=psh01com1hcom26

umask 007
module load miniconda/3
source activate recon

python normalize.py -d /gpfs/Labs/Cheng/phenome/COVID_Influenza_Progression/AAA589/data/24keV_center_section_7_recon_p3/ -o /gpfs/Labs/Cheng/phenome/COVID_Influenza_Progression/AAA589/data/24keV_center_section_7_recon_p3_sn/ -n 20.0,99.0
