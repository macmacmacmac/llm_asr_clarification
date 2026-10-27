#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=128g
#SBATCH -J "GoldGen"
#SBATCH -p long
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH -o det5.out
#SBATCH -e det5.out

nvidia-smi

# Load CUDA toolkit so nvcc is available for flashinfer JIT compilation
# TODO: Use 12.9 for RTX6000B
module load cuda/12.9.0

# Add CUDA libraries to LD_LIBRARY_PATH so Python can load the JIT-compiled flashinfer kernels
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Only deactivate if a venv is currently active
if declare -f deactivate > /dev/null; then deactivate; fi
source .vllm_venv/bin/activate

echo "Starting vLLM Offline Bulk Pipeline..."

# python -u -m run_scripts --scripts importance_detection.gold_labels_gen_script \
#     --split train \
#     --which-half full \
#     --window 1 \
#     --stride 1


python -u -m run_scripts --scripts importance_detection.generate_logp_deltas \
    --model Qwen/Qwen3-32B-FP8 \
    --split train \
    "$@"


echo "Pipeline completed."