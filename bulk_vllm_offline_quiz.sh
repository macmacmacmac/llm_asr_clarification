#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -J "vLLM_Offline"
#SBATCH -p long
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH -o offline_vllm.out
#SBATCH -e offline_vllm.out

nvidia-smi

# Load CUDA toolkit so nvcc is available for flashinfer JIT compilation
module load cuda/12.9.0

# Add CUDA libraries to LD_LIBRARY_PATH so Python can load the JIT-compiled flashinfer kernels
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Only deactivate if a venv is currently active
if declare -f deactivate > /dev/null; then deactivate; fi
source .vllm_venv/bin/activate

echo "Starting vLLM Offline Bulk Pipeline..."

# Earlier runs (already finished)
# python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
#     --answering-model "Qwen/Qwen3-32B-FP8" \
#     --scoring-model "Qwen/Qwen3-32B-FP8" \
#     --transcript-subdir rf-mis_lstmmask-imp \
#     --importance-detectors lstmmask \
#     --mistranscript-detectors ALL RF

# python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
#     --answering-model "Qwen/Qwen3-32B-FP8" \
#     --scoring-model "Qwen/Qwen3-32B-FP8" \
#     --transcript-subdir rf-mis_lstmmask-imp \
#     --importance-detectors lstmmask \
#     --mistranscript-detectors RF \
#     --score-op multiply

# python -u -m run_scripts --scripts quiz.single_vllm_offline_quiz \
#     --ami-path ./shared/datasets/amicorpus/validation \
#     --transcript-file parsed_diarized_gt \
#     --answering-model Qwen/Qwen3-32B-FP8 \
#     --scoring-model Qwen/Qwen3-32B-FP8

# python -u -m run_scripts --scripts quiz.single_vllm_offline_quiz \
#     --ami-path ./shared/datasets/amicorpus/validation \
#     --transcript-file custom_transcript_gt_segments \
#     --answering-model Qwen/Qwen3-32B-FP8 \
#     --scoring-model Qwen/Qwen3-32B-FP8

# python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
#     --answering-model "Qwen/Qwen3-32B-FP8" \
#     --scoring-model "Qwen/Qwen3-32B-FP8" \
#     --transcript-subdir rf-mis_cossim-imp

# Rebuilt transcripts, made from the current ASR transcript (already finished)
# python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
#     --answering-model "Qwen/Qwen3-32B-FP8" \
#     --scoring-model "Qwen/Qwen3-32B-FP8" \
#     --transcript-subdir rf-mis_cossim-imp_v2 \
#     --mistranscript-detectors RF GT ALL RANDOM

# Oracle transcripts: lines picked directly from the saved labels (already finished)
# python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
#     --answering-model "Qwen/Qwen3-32B-FP8" \
#     --scoring-model "Qwen/Qwen3-32B-FP8" \
#     --transcript-subdir oracle-labels \
#     --importance-detectors oracleswap oracleswapstd oraclemask \
#     --mistranscript-detectors ALL

# LLM line labels: N=50 clarifies every cited line (no meeting has more than 50)
python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
<<<<<<< HEAD
    --answering-model "Qwen/Qwen3-32B-FP8" \
    --scoring-model "Qwen/Qwen3-32B-FP8" \
    --transcript-subdir oracle-labels \
    --importance-detectors oracleattr \
    --mistranscript-detectors ALL \
    --num-lines 50
=======
    --answering-model "Qwen/Qwen3-4B" \
    --scoring-model "Qwen/Qwen3-8B"
>>>>>>> main

echo "Pipeline completed."
