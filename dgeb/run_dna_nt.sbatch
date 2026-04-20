#!/bin/bash
#SBATCH --job-name=dna_nt_v2_50m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

cd /scratch/users/khoang99/repos/BioSeqFeat/dgeb

export HF_HOME=/scratch/users/khoang99/hf_cache
# Note: remove TRANSFORMERS_OFFLINE=1 if the model is not yet cached locally.
# To pre-download: micromamba run -n dgeb_env python -c "
#   from transformers import AutoTokenizer, AutoModel
#   AutoTokenizer.from_pretrained('InstaDeepAI/nucleotide-transformer-v2-50m-multi-species', trust_remote_code=True)
#   AutoModel.from_pretrained('InstaDeepAI/nucleotide-transformer-v2-50m-multi-species', trust_remote_code=True)"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

micromamba run -n dgeb_env python bioseqfeat_dna_nt.py
