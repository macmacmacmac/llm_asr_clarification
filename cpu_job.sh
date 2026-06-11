#!/usr/bin/env bash 
#SBATCH -N 1                    
#SBATCH -n 1
#SBATCH -c 1 
#SBATCH --mem=8g                
#SBATCH -J "llm asr job"    
#SBATCH -p short                
#SBATCH -t 120          

srun --unbuffered python -m run_scripts "$@"