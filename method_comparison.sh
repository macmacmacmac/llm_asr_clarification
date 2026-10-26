#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -J "MethodCompare"
#SBATCH -p short
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B
#SBATCH -o method_comparison.out

# Quick-and-dirty teacher (importance) metric comparison on the meetings with teacher labels.
# 1. pipelinev4 writes custom_transcript_gt_segments_all_<imp>_<N>_clarify4.txt per meeting
#    (mistranscription detector ALL => lines ranked purely by the importance detector).
# 2. quiz.bulk_vllm_offline_quiz answers + grades those transcripts with Qwen3-32B-FP8 (greedy, the same
#    setup as the official baseline) and writes answer_using_* / score_using_* into vllm_quiz.json.
# 3. results/sampled_deltas/compare_methods.py summarizes into results/sampled_deltas/method_comparison.md.
# 32B-FP8 weights (~33 GB) + 40k context: RTX6000B (96 GB) fits comfortably, an L40S would be tight.

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

ALL_MEETINGS="ES2005d IN1012 IS1007c TS3005d"      # logp 8B/32B, cosine, random
SAMPLED_MEETINGS="ES2005d IN1012 TS3005d"           # sampled deltas only exist for these
NUM_LINES="10 20 30 40 50"

clarify() {  # clarify <importance detector> <seed> <output tag> <meetings...>
    local imp=$1 seed=$2 tag=$3; shift 3
    for n in $NUM_LINES; do
        python -u -m run_scripts --scripts clarifications.pipelinev4 \
            --ami_path ./shared/datasets/amicorpus/train \
            --meetings "$@" \
            --mistranscription-detector ALL \
            --importance-detector "$imp" \
            --num_lines "$n" \
            --seed "$seed" \
            --output-tag "$tag" \
            --sampled-scores-dir ./shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7
    done
}

clarify GT 47 "" $ALL_MEETINGS
clarify LOGP8B 47 "" $ALL_MEETINGS
clarify LOGP32B 47 "" $ALL_MEETINGS
clarify SAMPLED 47 "" $SAMPLED_MEETINGS
clarify RANDOM 47 "" $ALL_MEETINGS
clarify RANDOM 1 "_s1" $ALL_MEETINGS
clarify RANDOM 2 "_s2" $ALL_MEETINGS

# Missing transcripts (SAMPLED on IS1007c) are skipped with a warning by the quiz script.
python -u -m run_scripts --scripts quiz.bulk_vllm_offline_quiz \
    --answering-model "Qwen/Qwen3-32B-FP8" \
    --scoring-model "Qwen/Qwen3-32B-FP8" \
    --ami-path ./shared/datasets/amicorpus \
    --splits train \
    --meetings $ALL_MEETINGS \
    --mistranscript-detectors ALL \
    --importance-detectors GT LOGP8B LOGP32B SAMPLED RANDOM RANDOM_S1 RANDOM_S2 \
    --clarification-num-lines $NUM_LINES

python results/sampled_deltas/compare_methods.py

echo "Pipeline completed."
