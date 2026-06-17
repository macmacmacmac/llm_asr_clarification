import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import ipdb
import random
from llm_asr_clarification.models import OracleTranscript, OpenAIWrapper
from typing import Tuple, List
import re

SAMPLING_RATE = 16_000
GROUND_TRUTH_TRANSCRIPT = "parsed_diarized_gt.txt"


def extract_timestamps(line):
    # Extract all sequences of digits from the string
    numbers = re.findall(r'\d+', line)

    # Get the first two numbers and convert them to integers
    start_time = int(numbers[0])
    end_time = int(numbers[1])

    return start_time, end_time


# ┌───────────────────────────────────────────────┐
# │         CLARIFICATION CHOICE STRATEGY         │
# └───────────────────────────────────────────────┘
def choose_random_line(idx_pair: Tuple, transcript_lines: List[str]):
    idx = random.choice(idx_pair)
    chosen_line = transcript_lines[idx]
    start_time, end_time = extract_timestamps(chosen_line)
    return idx, start_time, end_time


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--transcript-file", type=str, default="whisper_tiny_diarized_transcript.txt")
    parser.add_argument("--meeting-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=47)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRANSCRIPT_FILE = args.transcript_file
    SEED = args.seed

    random.seed(SEED)


    # Global Variables
    OPENAI_MODEL = OpenAIWrapper()
    
    # Init Logger
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
    if args.meeting_name:
        meeting_folders=[DATASET_PATH / args.meeting_name]
    else:
        # Fetch all dataset meeting folders
        meeting_folders = [f for f in DATASET_PATH.iterdir() 
                            if (f.is_dir() and 
                                f.name not in ["ami_public_manual_1.6.2", "xinlu_data"])]

    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
            logger.info(f"Processing meeting {meeting_folder.name}")

            transcript_path = meeting_folder / "transcripts" / TRANSCRIPT_FILE
            oracle_transcript_path = meeting_folder / "transcripts" / GROUND_TRUTH_TRANSCRIPT

            # Fetch the Oracle Transcript
            oracle_transcript = OracleTranscript(
                transcript_path=oracle_transcript_path,
                logger=logger
            )

            with open(transcript_path, "r") as f:
                transcript_content = f.read()

            transcript_lines = transcript_content.split("\n")

            random_line_idxs = random.choices(range(0, len(transcript_lines)), k = 20)
            
            random_idx_pairs = list(zip(random_line_idxs[::2], random_line_idxs[1::2]))

            for idx_pair in random_idx_pairs:


                chosen_idx = random.choice(idx_pair)
                chosen_line = transcript_lines[chosen_idx]
                start_time, end_time = extract_timestamps(chosen_line)
                

                oracle_lines = oracle_transcript.get_oracle_transcription(start_time=start_time, end_time=end_time)


                oracle_lines = [f"({start_time} - {end_time}){line}" for line in oracle_lines]

                transcript_lines[chosen_idx] = "".join(oracle_lines).strip()
                logger.info(f"Clarified timestamps: {start_time} - {end_time}")



            # ┌───────────────────────────────────────────────┐
            # │                     SAVE                      │
            # └───────────────────────────────────────────────┘
            clarified_file_name = TRANSCRIPT_FILE.split(".")[0] + "_random_clarify.txt"
            fixed_transcript_file_path = meeting_folder / "transcripts" / clarified_file_name
            with open(fixed_transcript_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(transcript_lines))
                
            logger.info(f"Saved transcript for {fixed_transcript_file_path}\n\n")










            