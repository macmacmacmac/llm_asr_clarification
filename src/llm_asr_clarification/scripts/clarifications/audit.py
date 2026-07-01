import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import ipdb
import random
from llm_asr_clarification.models import OpenAIWrapper
from llm_asr_clarification.constants import SAMPLE_MEETINGS
import json
from typing import List
import re


GROUND_TRUTH_TRANSCRIPT = "parsed_diarized_gt.txt"
QUIZ_FILE = "quiz_from_parsed_diarized_gt.json"



TIMEWINDOW_FINDER_SYSTEM_PROMPT = """
# TASK DESCRIPTION
You are an expert at analyzing meeting transcripts. You will be given a meeting quiz question, its correct answer, and a chronological Ground Truth meeting transcript. 
Your task is to identify the exact line or lines in the transcript where the correct answer is explicitly discussed.

# TRANSCRIPT FORMAT
Every line follows this exact structure:
(START_TIME - END_TIME)[SPEAKER_ID]: Spoken Text...

# EXTRACTION RULES
1. Locate every line in the transcript that directly provides the factual basis for the provided answer.
2. You must copy the matching lines completely and VERBATIM. Do not alter the timestamps, speaker IDs, or text.

# OUTPUT FORMAT
Your response must be a valid JSON array of strings, and NOTHING else. Do not wrap the JSON in markdown code blocks. Each string in the array must be an exact copy of a matching transcript line.

Example Output:
[
  "(1011 - 1016)[Speaker X]: Lorem Ipsum ..."
]
"""

TIMEWINDOW_FINDER_USER_PROMPT_TEMPLATE = """
# QUIZ QUESTION
{quiz_question}

# CORRECT ANSWER
{correct_answer}

# GROUND TRUTH TRANSCRIPT
{ground_truth_transcript}

Output:
"""


MD_FILE_ITEM_TEMPLATE = """
## QUESTION
{quiz_question}

### CORRECT ANSWER
{correct_answer}

### CORRESPONDING LINES
#### GROUND TRUTH
```
{ground_truth_text}
```

#### WHISPER
```
{whisper_text}
```

---
"""


def extract_timestamps(line):
    # Extract all sequences of digits from the string
    numbers = re.findall(r'\d+', line)

    # Get the first two numbers and convert them to integers
    start_time = int(numbers[0])
    end_time = int(numbers[1])

    return start_time, end_time



def find_relevant_lines_using_ground_truth(transcript: str, gt_lines: List[str]):
    transcript_lines = transcript.split("\n")
    overlapping_lines = []
    for line in transcript_lines:
        # try:
        #     start, end = extract_timestamps(line)
        # except Exception as ex:
        #     ipdb.set_trace()
        #     continue

        start, end = extract_timestamps(line)

        for gt_line in gt_lines:
            gt_start, gt_end = extract_timestamps(gt_line)
            if start < gt_end and gt_start < end:
                overlapping_lines.append(line)
                break

    return overlapping_lines



# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--transcript-file", type=str, default="whisper_tiny_diarized_transcript.txt")
    parser.add_argument("--do-sample-meetings", action="store_true")
    parser.add_argument("--meeting-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=47)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRANSCRIPT_FILE = args.transcript_file
    SEED = args.seed

    random.seed(SEED)

    # Global Variables
    global TIMEWINDOW_FINDER_LLM
    TIMEWINDOW_FINDER_LLM = OpenAIWrapper(
        system_prompt=TIMEWINDOW_FINDER_SYSTEM_PROMPT
    )

    
    # Init Logger
    global logger
    logger = get_logger(exp_name)    
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    # Log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(
        f"Received the following arguments:\n{received_args_log}"
    )


    # ┌───────────────────────────────────────────────┐
    # │                   LOAD DATA                   │
    # └───────────────────────────────────────────────┘
    # Fetch a single meeting
    if args.meeting_name:
        meeting_folders=[DATASET_PATH / args.meeting_name]

    # Fetch only sample meetings
    elif args.do_sample_meetings:
        meeting_folders = [f for f in DATASET_PATH.iterdir()
                          if (f.is_dir() and
                              f.name in SAMPLE_MEETINGS)]
    # Fetch all meetings
    else:
        meeting_folders = [f for f in DATASET_PATH.iterdir() 
                            if (f.is_dir() and 
                                f.name not in ["ami_public_manual_1.6.2", "xinlu_data"])]
        

    # ┌───────────────────────────────────────────────┐
    # │               PROCESS MEETINGS                │
    # └───────────────────────────────────────────────┘
    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
            logger.info(f"Processing meeting {meeting_folder.name}")

            transcript_path = meeting_folder / "transcripts" / TRANSCRIPT_FILE
            gt_transcript_path = meeting_folder / "transcripts" / GROUND_TRUTH_TRANSCRIPT
            quiz_path = meeting_folder / "quiz" / QUIZ_FILE

            # Load all files
            with open(transcript_path, "r") as f:
                whisper_transcript = f.read().strip()

            with open(gt_transcript_path, "r") as f:
                gt_transcript = f.read().strip()

            with open(quiz_path, "r") as f:
                quiz = json.load(f)

            # Identify questions where generated transcripts failed
            failed_by_whisper = [q for q in quiz 
                                 if q[f"score_using_{TRANSCRIPT_FILE.split(".")[0]}"] != 1]
            failed_by_gt = [q for q in quiz 
                            if q[f"score_using_{GROUND_TRUTH_TRANSCRIPT.split(".")[0]}"] != 1]
            
            md_file_content = """# FAILED QUIZ BY WHISPER
            """
            
            # Using an LLM, find the timestamp window in the GT transcript, related to the correct answer
            for failed_question in failed_by_whisper:
                question = failed_question["question"]
                correct_answer = failed_question["correct_answer"]

                # Prepare user prompt
                user_prompt = TIMEWINDOW_FINDER_USER_PROMPT_TEMPLATE.format(
                    quiz_question = question,
                    correct_answer = correct_answer,
                    ground_truth_transcript = gt_transcript
                )

                # Prompt the LLM
                response = TIMEWINDOW_FINDER_LLM.prompt_chatgpt(
                    model = "gpt-5.4-mini",
                    prompt = user_prompt,
                    max_completion_tokens=1024
                )
                
                # Parse response as JSON
                try:
                    relevant_gt_lines = json.loads(response)
                except Exception as ex:
                    logger.info(f"Error while parsing {response} as JSON!")
                    continue

                # Find corresponding lines from Whisper transcript
                relevant_whisper_lines = find_relevant_lines_using_ground_truth(whisper_transcript, relevant_gt_lines)

                # Append to md_file_content
                failed_question_analysis_md = MD_FILE_ITEM_TEMPLATE.format(
                    quiz_question = question,
                    correct_answer = correct_answer,
                    ground_truth_text = "\n".join(relevant_gt_lines),
                    whisper_text = "\n".join(relevant_whisper_lines)
                )
                md_file_content += failed_question_analysis_md

            # Save the MD file
            md_file_path = meeting_folder / "quiz" / "failed_quiz_by_whisper.md"
            with open(md_file_path, "w") as f:
                f.write(md_file_content)
