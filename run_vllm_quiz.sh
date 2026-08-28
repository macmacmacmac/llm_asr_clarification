#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=32g
#SBATCH -J "vLLM"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH -C A100-80G
##SBATCH -C RTX6000B
#SBATCH -o vllm.out
#SBATCH -e vllm.out

# -----------------------------
# Run the Job (Example: Python Script / Module)
# -----------------------------
nvidia-smi

# Load CUDA toolkit so nvcc is available for flashinfer JIT compilation
module load cuda/12.8.0

# Add CUDA libraries to LD_LIBRARY_PATH so Python can load the JIT-compiled flashinfer kernels
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Only deactivate if a venv is currently active
if declare -f deactivate > /dev/null; then deactivate; fi
source .vllm_venv/bin/activate

START_TIME=$(date +%s)

python run_scripts.py --scripts quiz.vllm_answerer quiz.vllm_scorer \
    --ami-path ./shared/datasets/amicorpus/train \
    --transcript-file custom_transcript_gt_segments \
    --do-all-meetings \
    --model-to-use "Qwen/Qwen3-32B-FP8" \
    --max-model-len 40960 \
    --tensor-parallel-size 4

END_TIME=$(date +%s)
echo "Total execution time: $((END_TIME - START_TIME)) seconds"