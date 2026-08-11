import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import ipdb
import random
from typing import Tuple, List
import numpy as np
import re
import json

# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/amicorpus/validation")
    parser.add_argument("--transcript-file", type=str, default="custom_transcript_gt_segments.txt")

    parser.add_argument("--seed", type=int, default=47)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRANSCRIPT_FILE = args.transcript_file
    GT_FILE = "parsed_diarized_gt.txt"
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
    meeting_folders = [f for f in DATASET_PATH.iterdir() if f.is_dir()]

    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
            logger.info(f"Processing meeting {meeting_folder.name}")

            # Load the generated transcript
            transcript_path = meeting_folder / "transcripts" / TRANSCRIPT_FILE
            with open(transcript_path, "r") as f:
                transcript_content = f.read().strip()

            gen_lines = transcript_content.split("\n")

            # Load the GT transcript
            gt_transcript_path = meeting_folder / "transcripts" / GT_FILE
            with open(gt_transcript_path, "r") as f:
                gt_content = f.read().strip()
            gt_lines = gt_content.split("\n")

            # Load the beams artifact
            beam_results_2_path = meeting_folder / "artifacts" / "beam_results_2.json"
            with open(beam_results_2_path, "r", encoding="utf-8") as f:
                beam_results_2 = json.load(f)



            idxs_to_keep = []
            idxs_to_remove = []

            for i, line in enumerate(gt_lines):
                metadata, text_only = line.split(":")[:2]

                words = text_only.split(" ")
                num_words = len(words)

                numbers = [int(num) for num in re.findall(r'\d+', metadata)]
                len_time = max(numbers) - min(numbers)
                expected_num_words = len_time//2

                # ipdb.set_trace()

                if num_words < expected_num_words:
                    idxs_to_remove.append(i)
                else:
                    idxs_to_keep.append(i)
                
            # ipdb.set_trace()

            new_filter_gt = [gt_lines[i] for i in idxs_to_keep]
            # new_filter_gen = [gen_lines[i] for i in idxs_to_keep]
            new_beam_results_2 = [beam_results_2[i] for i in idxs_to_keep]
            # ┌───────────────────────────────────────────────┐
            # │                     SAVE                      │
            # └───────────────────────────────────────────────┘
            # with open(transcript_path, "w", encoding="utf-8") as f:
            #     f.write("\n".join(new_filter_gen))
                
            with open(gt_transcript_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_filter_gt))
                
            with open(beam_results_2_path, "w") as f:
                json.dump(new_beam_results_2, f, indent=4)










            