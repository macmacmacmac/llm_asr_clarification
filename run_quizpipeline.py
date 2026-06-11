import os


# transcript_files = ['qwen_transcript', 'tiny_transcript', 'large_transcript', 'whisper-large-v3_transcript', 'whisper-tiny_transcript']
transcript_files = ['qwen_transcript']#, 'tiny_transcript']
question_files = ['parsed_diarized_gt']

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
        command = f"sbatch -t 300 cpu_job.sh --scripts quiz_pipeline.question_answerer quiz_pipeline.question_scorer --transcript_file {t} --do_all_meetings --question_file {q} --model_to_use gpt-5.4-mini"
        os.system(command)


