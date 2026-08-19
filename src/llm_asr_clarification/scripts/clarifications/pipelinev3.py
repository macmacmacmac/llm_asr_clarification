import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import random
import re
import numpy as np
import ipdb

from llm_asr_clarification.models.MistranscriptionDetector import (
    RandomBernoulliDetector,
    GTDetector as MistranscriptionGTDetector,
    RFDetector,
    AllDetector
)
from llm_asr_clarification.models.ImportanceDetector import (
    RandomImportanceDetector,
    GTImportanceDetector,
    LSTMImportanceDetector
)

MIS_CLS_MAP = {
    'RANDOM' : RandomBernoulliDetector,
    'RF' : RFDetector,
    'GT' : MistranscriptionGTDetector,
    'ALL' : AllDetector
}

IMP_CLS_MAP = {
    'RANDOM' : RandomImportanceDetector,
    'GT' : GTImportanceDetector,
    'LSTM' : LSTMImportanceDetector
}


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

# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--clarify-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--mistranscription-detector", type=str, default="ALL")
    parser.add_argument("--importance-detector", type=str, default="GT")
    parser.add_argument("--seed", type=int, default=47)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRANSCRIPT_FILE = args.clarify_file
    GT_FILE = "parsed_diarized_gt.txt"
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    
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

    def get_mis_detector(detector_name, meeting_folder):
        return MIS_CLS_MAP[detector_name](meeting_path=meeting_folder)
        
    def get_imp_detector(detector_name, meeting_folder, transcript_file):
        return IMP_CLS_MAP[detector_name](meeting_path=meeting_folder, transcript_file=transcript_file)

    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
            logger.info(f"Processing meeting {meeting_folder.name}")

            # Load the generated transcript
            transcript_path = meeting_folder / "transcripts" / TRANSCRIPT_FILE
            with open(transcript_path, "r") as f:
                transcript_content = f.read().strip()

            original_transcript_lines = transcript_content.split("\n")
            updated_transcript_lines = original_transcript_lines.copy()
            num_lines = len(original_transcript_lines)

            # Load the GT transcript
            gt_transcript_path = meeting_folder / "transcripts" / GT_FILE
            with open(gt_transcript_path, "r") as f:
                gt_content = f.read().strip()
            gt_lines = gt_content.split("\n")
            
            line_numbers = list(range(num_lines))

            # Retrieve Mistranscription Detector
            mis_detector = get_mis_detector(args.mistranscription_detector, meeting_folder)
            mis_preds_bool_mask = mis_detector.pred_mistranscribed(line_numbers)
            num_mistranscribed = sum(mis_preds_bool_mask)
            logger.info(f"Mistranscription detector predicted: {num_mistranscribed} mistranscribed lines out of {num_lines} lines")

            # Retrieve Importance Detector
            imp_detector = get_imp_detector(args.importance_detector, meeting_folder, TRANSCRIPT_FILE)
            imp_preds_bool_mask = imp_detector.get_important_lines(line_numbers)
            num_important = sum(imp_preds_bool_mask)
            logger.info(f"Importance detector predicted: {num_important} important lines out of {num_lines} lines")

            # Intersect masks
            imp_line_idxs = [i for i in line_numbers if mis_preds_bool_mask[i] and imp_preds_bool_mask[i]]
            logger.info(f"Intersection resulted in {len(imp_line_idxs)} lines to clarify")

            # ipdb.set_trace()

            # For each idx 
            for chosen_idx in imp_line_idxs:
                chosen_line = original_transcript_lines[chosen_idx]
                updated_transcript_lines[chosen_idx] = gt_lines[chosen_idx]

                # Extract timestamps for the chosen line
                start_time, end_time = extract_timestamps(chosen_line)

                logger.info(f"Clarified timestamps: {start_time} - {end_time}")

            # ┌───────────────────────────────────────────────┐
            # │                     SAVE                      │
            # └───────────────────────────────────────────────┘
            clarified_file_name = f"{TRANSCRIPT_FILE.split('.')[0]}_{args.mistranscription_detector.lower()}_{args.importance_detector.lower()}_clarify3.txt"
            fixed_transcript_file_path = meeting_folder / "transcripts" / clarified_file_name
            with open(fixed_transcript_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(updated_transcript_lines))
                
            logger.info(f"Saved transcript for {fixed_transcript_file_path}\n\n")

