#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -J "SampledDeltas"
#SBATCH -p short
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH -o sampled_deltas_pilot.out

# RTX6000B on purpose: decoding K samples per prompt against 11-28k-token transcripts is
# KV-bandwidth bound, and the RTX Pro 6000 has ~2x the bandwidth of an L40S.

# Submit from the repo root (or worktree root) you want to run: paths below are relative to it.
cd "$SLURM_SUBMIT_DIR"

nvidia-smi

# Same CUDA setup as generate_gold_labels.sh used on RTX6000B (nvcc for any FlashInfer JIT)
module load cuda/12.9.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .vllm_venv

# The env's editable install points at the main checkout; make this checkout's src win.
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# Seeded sampling goes through the PyTorch sampler; skip FlashInfer sampler kernels entirely.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "Starting sampled-correctness deltas pilot..."

# Meetings run in sorted order: ES2005d (smallest, ~7% of the cost) finishes first and its
# per-meeting timing in the log calibrates the rest. Finished meetings are skipped on resubmit.
python -u -m run_scripts --scripts importance_detection.generate_sampled_deltas \
    --ami-path ./shared/datasets/amicorpus \
    --split train \
    --meetings TS3005d IN1012 ES2005d \
    --num-samples 8

python -u -m run_scripts --scripts importance_detection.analyze_sampled_deltas \
    --results-dir ./shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7 \
    --ami-path ./shared/datasets/amicorpus \
    --split train \
    --logp-paths ./shared/datasets/logp_deltas/train_Qwen3-8B-FP8.pt ./shared/datasets/logp_deltas/train_32b.pt

echo "Pipeline completed."
