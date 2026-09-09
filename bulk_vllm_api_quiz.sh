#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32g
#SBATCH -J "vLLM_Bulk_API"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100-80G
#SBATCH -o bulk.out
#SBATCH -e bulk.out

nvidia-smi

# Load CUDA toolkit so nvcc is available for flashinfer JIT compilation
module load cuda/12.8.0

# Add CUDA libraries to LD_LIBRARY_PATH so Python can load the JIT-compiled flashinfer kernels
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Only deactivate if a venv is currently active
if declare -f deactivate > /dev/null; then deactivate; fi
source .vllm_venv/bin/activate

MODEL_TO_USE="Qwen/Qwen3-32B-FP8"

# ---------------------------------------------------------
# Configurations
# ---------------------------------------------------------
splits=("validation")
# splits=("train")
clarification_num_lines=("10" "20" "30" "40" "50")
importance_detectors=("LSTM" "GT" "ALL")
mistranscript_detectors=("RF" "GT" "ALL")
versions=("4")

# Calculate total combinations for progress tracking
TOTAL_ITERS=$(( ${#splits[@]} * ${#clarification_num_lines[@]} * ${#importance_detectors[@]} * ${#mistranscript_detectors[@]} * ${#versions[@]} ))
CURRENT_ITER=0

echo "Starting Bulk vLLM API job"
echo "Total configurations to run: $TOTAL_ITERS"

# Start the vLLM server in the background (we do this OUTSIDE the loop!)
echo "Starting vLLM API server in the background..."
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_TO_USE}" \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --seed 47 \
  --port 8000 &
VLLM_PID=$!

# Poll the server every 5 seconds until it responds
echo "Waiting for vLLM server to be ready..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 5
done
echo "vLLM server is up!"

START_TIME=$SECONDS

# Iterate over cartesian product
for split in "${splits[@]}"; do
  for clar_num_line in "${clarification_num_lines[@]}"; do
    for imp_det in "${importance_detectors[@]}"; do
      for mis_det in "${mistranscript_detectors[@]}"; do
        for version in "${versions[@]}"; do
          
          ((CURRENT_ITER++))
          
          # Calculate Elapsed and ETA
          ELAPSED=$(( SECONDS - START_TIME ))
          AVG_TIME=$(( ELAPSED / CURRENT_ITER ))
          REMAINING=$(( TOTAL_ITERS - CURRENT_ITER ))
          ETA=$(( AVG_TIME * REMAINING ))
          
          # Format times to MM:SS
          printf -v ELAPSED_FMT "%02d:%02d" $((ELAPSED/60)) $((ELAPSED%60))
          printf -v ETA_FMT "%02d:%02d" $((ETA/60)) $((ETA%60))
          
          # Convert strings to lowercase for the filename using bash parameter expansion
          imp_det_lower="${imp_det,,}"
          mis_det_lower="${mis_det,,}"
          
          transcript_file="custom_transcript_gt_segments_${mis_det_lower}_${imp_det_lower}_${clar_num_line}_clarify${version}"
          
          echo "=========================================================="
          echo "[Progress: $CURRENT_ITER / $TOTAL_ITERS] | Elapsed: $ELAPSED_FMT | ETA: $ETA_FMT"
          echo "Running configuration: $transcript_file on split $split"
          echo "=========================================================="
          
          python -u -m run_scripts \
            --scripts quiz.vllm_api_answerer quiz.vllm_api_scorer \
            --ami-path "./shared/datasets/amicorpus/${split}" \
            --transcript-file "${transcript_file}" \
            --do-all-meetings \
            --model-to-use "${MODEL_TO_USE}"
            
        done
      done
    done
  done
done

# Cleanup
echo "All $TOTAL_ITERS configurations completed in $(( SECONDS - $START_TIME )) seconds."
echo "Killing vLLM server..."
kill $VLLM_PID
echo "Done."
