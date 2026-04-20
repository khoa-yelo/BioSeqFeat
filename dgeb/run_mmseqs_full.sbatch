#!/bin/bash
#SBATCH --job-name=bsf_mmseqs_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs

cd /scratch/users/khoang99/repos/BioSeqFeat

export HF_HOME=/scratch/users/khoang99/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Step 1: sample 1000 anchors from existing 89k representatives ---
echo "[$(date)] Sampling 1000 anchors..."
python db/sample_anchors.py \
    --input db/bacteria_anchors.fasta \
    --output db/bacteria_anchors_1k.fasta \
    --n 1000 --seed 42
echo "[$(date)] Anchor set: 1000 representative sequences"

# --- Step 2: MMseqs2-only landmark embedding ---
echo "[$(date)] Running MMseqs2-only evaluation..."
cd dgeb
python bioseqfeat_mmseqs.py \
    --anchor-fasta ../db/bacteria_anchors_1k.fasta \
    --cpu 8

# --- Step 3: MMseqs2 + BLOSUM + HMM combined ---
echo "[$(date)] Running MMseqs2+BLOSUM+HMM combined evaluation..."
python bioseqfeat_mmseqs.py \
    --anchor-fasta ../db/bacteria_anchors_1k.fasta \
    --combined \
    --cpu 8

echo "[$(date)] All done."
