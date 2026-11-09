"""
Clarify all validation meetings using the masked-label importance model.
Run from the project root on a GPU node: python run_clarify_lstmmask.py
Edit the settings below instead of passing arguments. Stops at the first error.
"""

from llm_asr_clarification.scripts.clarifications import pipelinev4

SPLIT = "validation"
MODEL_PATH = "./shared/model_weights/mask_classifier/pos5_neg5/importance_lstm.pt"
RUN_NAME = "lstmmask"
OUTPUT_SUBDIR = "rf-mis_lstmmask-imp"
NUM_LINES = [10, 20, 30, 40, 50]

# (mistranscription detector, how to combine its score with the importance score)
# ALL has no multiply run: its score is always 1, so both ops rank lines the same way.
DETECTOR_CONFIGS = [
    # Already finished
    # ("ALL", "add"),
    # ("RF", "add"),
    # ("RF", "multiply"),
    ("GT", "add"),
]

runs = [(num_lines, detector, op) for num_lines in NUM_LINES for detector, op in DETECTOR_CONFIGS]

for i, (num_lines, detector, op) in enumerate(runs, start=1):
    print(f"\n[{i}/{len(runs)}] num_lines={num_lines}, mistranscription detector={detector}, score op={op}", flush=True)
    pipelinev4.run([
        "--ami_path", f"./shared/datasets/amicorpus/{SPLIT}",
        "--mistranscription-detector", detector,
        "--importance-detector", "LSTM",
        "--importance-model-path", MODEL_PATH,
        "--run-name", RUN_NAME,
        "--score-op", op,
        "--output-subdir", OUTPUT_SUBDIR,
        "--num_lines", str(num_lines),
    ])

print(f"\nFinished {len(runs)} runs. Clarified files are in transcripts/{OUTPUT_SUBDIR}/ for each meeting.")
