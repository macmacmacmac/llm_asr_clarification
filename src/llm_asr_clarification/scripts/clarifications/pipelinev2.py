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
import numpy as np
from llm_asr_clarification.constants import SAMPLE_MEETINGS
from llm_asr_clarification.models import MistranscriptionDetector
from llm_asr_clarification.models.MistranscriptionDetector import (
    RandomBernoulliDetector,
    GTDetector,
    RFDetector
)

SAMPLING_RATE = 16_000
GROUND_TRUTH_TRANSCRIPT = "parsed_diarized_gt.txt"


# ┌───────────────────────────────────────────────┐
# │                   HELPER METHODS              │
# └───────────────────────────────────────────────┘
def extract_timestamps(line):
    # Extract all sequences of digits from the string
    numbers = re.findall(r'\d+', line)

    # Get the first two numbers and convert them to integers
    start_time = int(numbers[0])
    end_time = int(numbers[1])

    return start_time, end_time

# def print_timestamps(idx_pair: Tuple[int], transcript_lines: List[str]):
#     logger.info("Processing indices")
#     for idx in idx_pair:
#         line = transcript_lines[idx]
#         start_time, end_time = extract_timestamps(line)
#         logger.info(f"idx: {idx}, line: {line}, start: {start_time}, end: {end_time}")
#     logger.info("---")



# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--transcript-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--do-sample-meetings", type=bool, default=False)
    parser.add_argument("--detectors", nargs="+", default=["RANDOM", "RF", "GT"])
    parser.add_argument("--seed", type=int, default=47)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRANSCRIPT_FILE = args.transcript_file
    GT_FILE = "parsed_diarized_gt.txt"
    global STRATEGY
    STRATEGY = args.strategy
    SEED = args.seed

    random.seed(SEED)
    
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
    if args.meeting_name:
        meeting_folders=[DATASET_PATH / args.meeting_name]
    else:
        # Fetch all dataset meeting folders
        # meeting_folders = [f for f in DATASET_PATH.iterdir() 
        #                     if (f.is_dir() and 
        #                         f.name not in ["ami_public_manual_1.6.2", "xinlu_data"])]
        meeting_folders = [f for f in DATASET_PATH.iterdir()
                          if (f.is_dir() and
                              f.name in SAMPLE_MEETINGS)]
        
    def clarify_meeting_using_detector(detector_name):
        # Wrap logging with tqdm
        with logging_redirect_tqdm(loggers=[logger]):

            # Process all Meeting Folders
            for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
                logger.info(f"Processing meeting {meeting_folder.name}")

                # Load the generated transcript
                transcript_path = meeting_folder / "transcripts" / TRANSCRIPT_FILE
                with open(transcript_path, "r") as f:
                    transcript_content = f.read()

                # Maintain two lists, one stores original line and one stores updated lines
                original_transcript_lines = transcript_content.split("\n")
                updated_transcript_lines = original_transcript_lines.copy()
                num_lines = len(original_transcript_lines)

                # Load the GT transcript
                gt_transcript_path = meeting_folder / "transcripts" / GT_FILE
                with open(gt_transcript_path, "r") as f:
                    gt_content = f.read()

                gt_lines = gt_content.split("\n")


                # Predict which generated lines are mistranscribed
                detector_cls_mapping = {
                    'RANDOM' : RandomBernoulliDetector,
                    'RF' : RFDetector,
                    'GT' : GTDetector
                }
                detector_cls = detector_cls_mapping[detector_name]

                ipdb.set_trace()
                
                detector = detector_cls(meeting_path=transcript_path)
                preds_bool_mask = detector.pred_mistranscribed(range(num_lines))
                logger.info(f"This detector predicted: {sum(preds_bool_mask)} mistranscribed lines out of {num_lines} lines")

                mistranscribed_lines_idxs = np.arange(num_lines)[preds_bool_mask]
                
                # Select 20 random lines out of the mistranscribed ones
                random_line_idxs = np.random.choice(mistranscribed_lines_idxs, 20, replace=False)

                # For each idx 
                for chosen_idx in random_line_idxs:
                    chosen_line = original_transcript_lines[chosen_idx]
                    updated_transcript_lines[chosen_idx] = gt_lines[chosen_idx]

                    # Extract timestamps for the chosen line
                    start_time, end_time = extract_timestamps(chosen_line)

                    logger.info(f"Clarified timestamps: {start_time} - {end_time}")

                # ┌───────────────────────────────────────────────┐
                # │                     SAVE                      │
                # └───────────────────────────────────────────────┘
                clarified_file_name = TRANSCRIPT_FILE.split(".")[0] + f"_{detector_name.lower()}_clarify.txt"
                fixed_transcript_file_path = meeting_folder / "transcripts" / clarified_file_name
                with open(fixed_transcript_file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(updated_transcript_lines))
                    
                logger.info(f"Saved transcript for {fixed_transcript_file_path}\n\n")

    for det_name in args.detectors:
        clarify_meeting_using_detector(det_name)











            