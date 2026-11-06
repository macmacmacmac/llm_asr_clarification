#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=16g
#SBATCH -J "transcribe"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C L40S
#SBATCH -o noise_val.out
#SBATCH -e noise_val.out

# -----------------------------
# Run the Job (Example: Python Script / Module)
# -----------------------------
python run_scripts.py --scripts transcription.custom.generate_diarized_gt_segments \
    --dataset-path "./shared/datasets/amicorpus/validation" \
    --add-noise