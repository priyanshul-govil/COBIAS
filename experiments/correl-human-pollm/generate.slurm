#!/bin/bash
#SBATCH --job-name="ratings-gen"
#SBATCH --mail-user=pgovil1@umbc.edu
#SBATCH --mem=50000
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx_8000"

conda init bash
source ~/.bashrc
conda activate cobias
huggingface-cli login --token "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"
# export TRANSFORMERS_CACHE="/scratch/$SLURM_JOBID"
cd correl-human-pollm
python evaluate.py $1

