#!/bin/bash -l
#SBATCH --job-name=ffmpeg
#SBATCH --output=../log/ffmpeg_%j_%N.std
#SBATCH --error=../log/ffmpeg_%j_%N.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=340G
#SBATCH --cpus-per-task=88

module load ffmpeg

ffmpeg -framerate 30 -start_number 1067 -probesize 2G -i /gpfs/Labs/Cheng/phenome/COVID_Influenza_Progression/AAA673/data/25keV_Patient10B_section5_recon_p23_denoise_8bit/CL_AAA673_%d.tif -c:v wmv2 /gpfs/Labs/Cheng/phenome/8bittestvid.wmv

