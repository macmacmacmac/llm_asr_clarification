#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -J "LogpRepro"
#SBATCH -p short
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH -o logp_repro.out

# Reproducibility check for the 32B logp deltas: compute ES2005d twice with identical settings
# (two separate processes, same code/model/args as logp_deltas_full.sh) into separate files, then compare
# with results/sampled_deltas/logp_repro_compare.py. ~2 x (16 min + 5 min load) on an RTX6000B.

cd "$SLURM_SUBMIT_DIR"

nvidia-smi

module load cuda/12.9.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .vllm_venv

export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p logs

# --max-num-batched-tokens 8192: the first attempt (job 2411784) crashed in both runs on a fully uncached
# 15.9k-token prompt, whose 15.9k x 152k-vocab prompt-logprob logits overflow cuBLAS int32 indexing.
for run in 1 2; do
    python -u -m run_scripts --scripts importance_detection.generate_logp_deltas \
        --model Qwen/Qwen3-32B-FP8 \
        --ami-path ./shared/datasets/amicorpus \
        --split train \
        --meetings ES2005d \
        --max-num-batched-tokens 8192 \
        --save-path "./shared/datasets/logp_deltas/full/repro/ES2005d_run${run}.pt"
    if [ ! -f "./shared/datasets/logp_deltas/full/repro/ES2005d_run${run}.pt" ]; then
        echo "Run $run produced no output; stopping." >&2
        exit 1
    fi
done

echo "Pipeline completed."
