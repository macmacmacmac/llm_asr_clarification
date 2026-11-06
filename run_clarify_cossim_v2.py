"""
Rebuild the earlier clarified transcripts from the current ASR transcript.
Run from the project root on a GPU node: python run_clarify_cossim_v2.py
Edit the settings below instead of passing arguments. Stops at the first error.
"""

from llm_asr_clarification.scripts.clarifications import pipelinev4

SPLIT = "validation"
OUTPUT_SUBDIR = "rf-mis_cossim-imp_v2"
NUM_LINES = [10, 20, 30, 40, 50]
IMPORTANCE_DETECTORS = ["LSTM", "GT", "ALL"]
MISTRANSCRIPTION_DETECTORS = ["RF", "GT", "ALL", "RANDOM"]

runs = [
    (num_lines, importance, mistranscription)
    for num_lines in NUM_LINES
    for importance in IMPORTANCE_DETECTORS
    for mistranscription in MISTRANSCRIPTION_DETECTORS
]

for i, (num_lines, importance, mistranscription) in enumerate(runs, start=1):
    print(f"\n[{i}/{len(runs)}] num_lines={num_lines}, importance detector={importance}, mistranscription detector={mistranscription}", flush=True)
    pipelinev4.run([
        "--ami_path", f"./shared/datasets/amicorpus/{SPLIT}",
        "--mistranscription-detector", mistranscription,
        "--importance-detector", importance,
        "--output-subdir", OUTPUT_SUBDIR,
        "--num_lines", str(num_lines),
    ])

print(f"\nFinished {len(runs)} runs. Clarified files are in transcripts/{OUTPUT_SUBDIR}/ for each meeting.")
