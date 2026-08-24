import os
import subprocess
from itertools import product

ARGUMENTS = {
    # 'transcript_files' : [
    #     # 'whisper_tiny_diarized_transcript',
    #     # 'whisper_tiny_diarized_transcript_random_clarify',
    #     # 'whisper_tiny_diarized_transcript_llm-orig-ctx_clarify',
    #     # 'whisper_tiny_diarized_transcript_llm-gt-ctx_clarify',
    #     # 'whisper_tiny_diarized_transcript_llm-orig-ctx_clarify_sample2',
    #     # 'whisper_tiny_diarized_transcript_llm-gt-ctx_clarify_sample2',
    #     # 'qwen_transcript',
    #     # 'tiny_transcript',
    #     # 'large_transcript',
    #     # 'whisper-large-v3_transcript',
    #     # 'whisper-tiny_transcript',
    #     # 'parsed_diarized_gt',
    #     'custom_transcript_gt_segments',
    #     # 'custom_transcript_gt_segments_gt_clarify',
    #     # 'custom_transcript_gt_segments_random_clarify',
    #     # 'custom_transcript_gt_segments_rf_clarify',
    #     # 'custom_transcript_gt_segments_clarify_only_importance',
    #     # 'custom_transcript_gt_segments_gt_clarify2',
    #     # 'custom_transcript_gt_segments_noise',
    #     # 'custom_transcript_gt_segments_all_gt_clarify3',
    #     # 'custom_transcript_gt_segments_all_lstm_clarify3',
    #     # 'custom_transcript_gt_segments_gt_lstm_clarify3',
    # ],
    'splits' : ['validation'],
    # 'seeds' : ["1", "2", "3", "4", "5"],
    'clarification_num_lines': ["10", "20", "30", "40", "50"],
    'importance_detectors' : ["LSTM", "GT"],
    'mistranscript_detectors' : ["RF", "GT", "ALL"]
}

MODEL_TO_USE = 'gpt-4o-mini' #'gpt-5.4-mini'

keys = ARGUMENTS.keys()
values = ARGUMENTS.values()

# 2. Compute the Cartesian product and rebuild dictionaries
combinations = [dict(zip(keys, v)) for v in product(*values)]

for arg in combinations:

    split = arg['splits']
    clarification_num_line = arg['clarification_num_lines']
    importance_detector = arg['importance_detectors']
    mistranscript_detector = arg['mistranscript_detectors']

    transcript_file = f"custom_transcript_gt_segments_{mistranscript_detector.lower()}_{importance_detector.lower()}_{clarification_num_line}_clarify3"

    command_args = [
        "sbatch",
        "-t", "60",
        "-o", f"quiz_{transcript_file}_{split}.out",
        "-e", f"quiz_{transcript_file}_{split}.out",
        "cpu_job.sh",
        "--scripts", "clarifications.pipelinev3", "quiz.question_answerer", "quiz.question_scorer",
        "--ami_path", f"./shared/datasets/amicorpus/{split}",
        "--num_lines", clarification_num_line,
        "--mistranscription-detector", mistranscript_detector,
        "--importance-detector", importance_detector,
        "--transcript_file", transcript_file,
        "--question_file", "parsed_diarized_gt",
        "--do_all_meetings",
        "--model_to_use", MODEL_TO_USE
    ]

    # Run the command
    subprocess.run(command_args, check=False)

