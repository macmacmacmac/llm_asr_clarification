import os
import subprocess


transcript_files = [
    # 'whisper_tiny_diarized_transcript',
    # 'whisper_tiny_diarized_transcript_random_clarify',
    # 'whisper_tiny_diarized_transcript_llm-orig-ctx_clarify',
    # 'whisper_tiny_diarized_transcript_llm-gt-ctx_clarify',
    # 'whisper_tiny_diarized_transcript_llm-orig-ctx_clarify_sample2',
    # 'whisper_tiny_diarized_transcript_llm-gt-ctx_clarify_sample2',
    # 'qwen_transcript',
    # 'tiny_transcript',
    # 'large_transcript',
    # 'whisper-large-v3_transcript',
    # 'whisper-tiny_transcript',
    # 'parsed_diarized_gt',
    # 'custom_transcript_gt_segments',
    # 'custom_transcript_gt_segments_gt_clarify',
    # 'custom_transcript_gt_segments_random_clarify',
    # 'custom_transcript_gt_segments_rf_clarify',
    # 'custom_transcript_gt_segments_clarify_only_importance',
    # 'custom_transcript_gt_segments_gt_clarify2',
    'custom_transcript_gt_segments_noise',
]
question_files = ['parsed_diarized_gt']
MODEL_TO_USE = 'gpt-4o-mini'
# MODEL_TO_USE = 'gpt-5.4-mini'


# for q in question_files:
#     command = f"sbatch -t 300 cpu_job.sh --scripts quiz_pipeline.question_generator --question_file {q} --do_all_meetings --model_to_use gpt-5.4-mini"
#     os.system(command)

# for t in transcript_files:
#     for q in question_files:
#         command = f"sbatch -t 300 cpu_job.sh --scripts quiz_pipeline.question_answerer quiz_pipeline.question_scorer --transcript_file {t} --do_all_meetings --question_file {q} --model_to_use gpt-5.4-mini"
#         os.system(command)

# for t in transcript_files:
#     for q in question_files:
#         command = f"sbatch -t 300 cpu_job.sh --scripts quiz_pipeline.question_scorer --transcript_file {t} --do_all_meetings --question_file {q} --model_to_use gpt-5.4-mini"
#         os.system(command)

for t in transcript_files:
    for q in question_files:
        # command = f"sbatch -t 30 -o quiz_{t}.out -e quiz_{t}.out cpu_job.sh \
        #     --scripts quiz.question_answerer quiz.question_scorer \
        #     --transcript_file {t} \
        #     --do_all_meetings \
        #     --question_file {q} \
        #     --model_to_use {MODEL_TO_USE}"
        
        # command = f"sbatch -t 30 -o quiz_{t}.out -e quiz_{t}.out cpu_job.sh \
        #     --scripts quiz.question_answerer quiz.question_scorer \
        #     --transcript_file {t} \
        #     --question_file {q} \
        #     --do_all_meetings \
        #     --model_to_use {MODEL_TO_USE}"
        # os.system(command)

        # Define the arguments clearly as a list
        command_args = [
            "sbatch",
            "-t", "60",
            "-o", f"quiz_{t}.out",
            "-e", f"quiz_{t}.out",
            "cpu_job.sh",
            "--scripts", "quiz.question_answerer", "quiz.question_scorer",
            "--transcript_file", t,
            "--question_file", q,
            "--do_all_meetings",
            "--model_to_use", MODEL_TO_USE
        ]

        # Run the command
        subprocess.run(command_args, check=False)

