#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -J "vLLM_Offline"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100-80G
#SBATCH -o offline_vllm.out
#SBATCH -e offline_vllm.out

nvidia-smi

# Load CUDA toolkit so nvcc is available for flashinfer JIT compilation
module load cuda/12.8.0

# Add CUDA libraries to LD_LIBRARY_PATH so Python can load the JIT-compiled flashinfer kernels
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Only deactivate if a venv is currently active
if declare -f deactivate > /dev/null; then deactivate; fi
source .vllm_venv/bin/activate

echo "Starting vLLM Offline Bulk Pipeline..."

python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
    --answering-model "Qwen/Qwen3-32B-FP8" \
    --scoring-model "Qwen/Qwen3-32B-FP8"

echo "Pipeline completed."
