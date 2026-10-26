#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -J "LogpValCompare"
#SBATCH -p short
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH -o logp32b_val_comparison.out

# Validation comparison of the logp 32B teacher vs cosine (pipelinev4's GT importance detector) vs random
# (ALL importance = all lines tied, random tie-break), all rerun fresh in one job with Qwen3-32B.
# Nothing existing is overwritten:
#   - transcripts get the tag "_r2": custom_transcript_gt_segments_all_<imp>_r2_<K>_clarify4.txt
#   - quiz fields are answer_using_/score_using_custom_transcript_gt_segments_all_<imp>_r2_<K>_clarify4
#   - each validation vllm_quiz.json is first copied to vllm_quiz.json.bak_20261022 (never overwritten)
# ~2,700 answer prompts (3 methods x 5 K x 18 meetings x 10 questions) at batch size 1: ~3 GPU-h.
# Needs the validation logp file written by logp_deltas_full.sh.

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

NUM_LINES="10 20 30 40 50"
TAG="_r2"
LOGP_PATH=./shared/datasets/logp_deltas/full/validation_Qwen3-32B-FP8.pt

if [ ! -f "$LOGP_PATH" ]; then
    echo "Missing $LOGP_PATH; run logp_deltas_full.sh validation first." >&2
    exit 1
fi

for quiz in ./shared/datasets/amicorpus/validation/*/quiz/vllm_quiz.json; do
    cp -n "$quiz" "$quiz.bak_20261022"
done

for imp in LOGP32B GT ALL; do
    for n in $NUM_LINES; do
        python -u -m run_scripts --scripts clarifications.pipelinev4 \
            --ami_path ./shared/datasets/amicorpus/validation \
            --mistranscription-detector ALL \
            --importance-detector "$imp" \
            --output-tag "$TAG" \
            --num_lines "$n" \
            --logp-path "$LOGP_PATH"
    done
done

python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
    --answering-model "Qwen/Qwen3-32B-FP8" \
    --scoring-model "Qwen/Qwen3-32B-FP8" \
    --ami-path ./shared/datasets/amicorpus \
    --splits validation \
    --mistranscript-detectors ALL \
    --importance-detectors LOGP32B_R2 GT_R2 ALL_R2 \
    --clarification-num-lines $NUM_LINES

python results/sampled_deltas/val_accuracy_table.py

echo "Pipeline completed."
