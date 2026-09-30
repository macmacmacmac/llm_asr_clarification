#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -J "GoldGen"
#SBATCH -p long
#SBATCH -t 5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C L40S
#SBATCH -o gold_window_1_second.out

nvidia-smi

# Load CUDA toolkit so nvcc is available for flashinfer JIT compilation
module load cuda/12.8.0

# Add CUDA libraries to LD_LIBRARY_PATH so Python can load the JIT-compiled flashinfer kernels
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Only deactivate if a venv is currently active
if declare -f deactivate > /dev/null; then deactivate; fi
source .vllm_venv/bin/activate

echo "Starting vLLM Offline Bulk Pipeline..."

python -u -m run_scripts --scripts importance_detection.gold_labels_gen_script \
    --split train \
    --which-half second \
    --window 1 \
    --stride 1

echo "Pipeline completed."