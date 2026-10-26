#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=128g
#SBATCH -J "LogpDeltas32B"
#SBATCH -p long
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B

# Full-dataset logp deltas with Qwen3-32B-FP8 (the teacher picked from the pilot comparison).
# Usage: sbatch -o <log> logp_deltas_full.sh <split> <num_shards> <shard_idx> [extra script args, e.g. --meetings EN2001e]
# Set SAVE_PATH in the environment (sbatch --export=ALL,SAVE_PATH=...) to override the output file.
# Each shard writes its own file, so shards never write the same .pt; merge them afterwards with
#   python -m run_scripts --scripts importance_detection.merge_logp_shards --split <split>
# Finished meetings are skipped on resubmit (the script saves after every meeting).
# --mem=128g: the script tokenizes every (version, question) prompt of a meeting at once (~15 GB of
# Python ints for the longest meetings), as in the earlier logp runs.

SPLIT=$1
NUM_SHARDS=${2:-1}
SHARD_IDX=${3:-0}

cd "$SLURM_SUBMIT_DIR"

nvidia-smi

module load cuda/12.9.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .vllm_venv

# The env's editable install points at the main checkout; make this checkout's src win.
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p logs

if [ -n "$SAVE_PATH" ]; then
    :  # caller-provided (e.g. a calibration run on one meeting)
elif [ "$NUM_SHARDS" -gt 1 ]; then
    SAVE_PATH="./shared/datasets/logp_deltas/full/${SPLIT}_Qwen3-32B-FP8_shard${SHARD_IDX}of${NUM_SHARDS}.pt"
else
    SAVE_PATH="./shared/datasets/logp_deltas/full/${SPLIT}_Qwen3-32B-FP8.pt"
fi

echo "Starting logp deltas: split=$SPLIT shard=$SHARD_IDX/$NUM_SHARDS -> $SAVE_PATH"

python -u -m run_scripts --scripts importance_detection.generate_logp_deltas \
    --model Qwen/Qwen3-32B-FP8 \
    --ami-path ./shared/datasets/amicorpus \
    --split "$SPLIT" \
    --num-shards "$NUM_SHARDS" \
    --shard-idx "$SHARD_IDX" \
    --save-path "$SAVE_PATH" \
    "${@:4}"

echo "Pipeline completed."
