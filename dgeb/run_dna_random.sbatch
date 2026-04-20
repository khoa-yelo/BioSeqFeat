#!/bin/bash
#SBATCH --job-name=dna_random
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

cd /scratch/users/khoang99/repos/BioSeqFeat/dgeb

export HF_HOME=/scratch/users/khoang99/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

micromamba run -n dgeb_env python bioseqfeat_dna_random.py
