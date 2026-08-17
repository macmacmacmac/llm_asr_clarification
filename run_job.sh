#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=16g
#SBATCH -J "ASRNoise"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -o custom.out
#SBATCH -e custom.out

# -----------------------------
# Run the Job (Example: Python Script / Module)
# -----------------------------
python run_scripts.py --scripts transcription.custom.generate_diarized_gt_segments --add-noise