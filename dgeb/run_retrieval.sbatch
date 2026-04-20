#!/bin/bash
#SBATCH --job-name=bsf_retrieval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

cd /scratch/users/khoang99/repos/BioSeqFeat/dgeb

export HF_HOME=/scratch/users/khoang99/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python bioseqfeat_retrieval.py
