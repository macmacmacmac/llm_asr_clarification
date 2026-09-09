#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32g
#SBATCH -J "vLLM_API_Test"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100-80G
#SBATCH -o vllm_api.out
#SBATCH -e vllm_api.out

nvidia-smi

# Load CUDA toolkit so nvcc is available for flashinfer JIT compilation
module load cuda/12.8.0

# Add CUDA libraries to LD_LIBRARY_PATH so Python can load the JIT-compiled flashinfer kernels
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Only deactivate if a venv is currently active
if declare -f deactivate > /dev/null; then deactivate; fi
source .vllm_venv/bin/activate

# Start the vLLM server in the background
echo "Starting vLLM API server in the background..."
python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-32B-FP8" \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --seed 47 \
  --port 8000 &
VLLM_PID=$!

# Poll the server every 10 seconds until it responds
echo "Waiting for vLLM server to be ready..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    echo "Still waiting for vLLM server..."
    sleep 10
done
echo "vLLM server is up!"

START_TIME=$SECONDS

# Run the single pipeline using the new API scripts
python -u -m run_scripts \
    --scripts quiz.vllm_api_answerer quiz.vllm_api_scorer \
    --ami-path "./shared/datasets/amicorpus/validation" \
    --transcript-file "custom_transcript_gt_segments" \
    # --do-all-meetings \
    --model-to-use "Qwen/Qwen3-32B-FP8"

# Run the single pipeline using the new API scripts
python -u -m run_scripts \
    --scripts quiz.vllm_api_answerer quiz.vllm_api_scorer \
    --ami-path "./shared/datasets/amicorpus/validation" \
    --transcript-file "parsed_diarized_gt" \
    # --do-all-meetings \
    --model-to-use "Qwen/Qwen3-32B-FP8"

echo "Took $(( SECONDS - $START_TIME )) seconds."

# Cleanup
echo "Pipeline finished. Killing vLLM server..."
kill $VLLM_PID
echo "Done."
