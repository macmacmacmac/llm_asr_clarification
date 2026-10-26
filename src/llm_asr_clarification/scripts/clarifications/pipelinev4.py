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
from functools import partial

from llm_asr_clarification.models.MistranscriptionDetector import (
    RandomBernoulliDetector,
    GTDetector as MistranscriptionGTDetector,
    RFDetector,
    AllDetector
)
from llm_asr_clarification.models.ImportanceDetector import (
    RandomImportanceDetector,
    GTImportanceDetector,
    LSTMImportanceDetector,
    ALLImportanceDetector,
    SampledImportanceDetector,
    LogpImportanceDetector
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
    'LSTM' : LSTMImportanceDetector,
    'ALL' : ALLImportanceDetector,
    'SAMPLED' : SampledImportanceDetector,
    'LOGP8B' : partial(LogpImportanceDetector, logp_path="./shared/datasets/logp_deltas/train_Qwen3-8B-FP8.pt"),
    'LOGP32B' : partial(LogpImportanceDetector, logp_path="./shared/datasets/logp_deltas/train_32b.pt")
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
    parser.add_argument("--ami_path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--clarify-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--mistranscription-detector", type=str, default="ALL")
    parser.add_argument("--importance-detector", type=str, default="LSTM")
    parser.add_argument("--num_lines", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--meetings", type=str, nargs="+", default=None, help="Only process these meetings (default: all in --ami_path)")
    parser.add_argument("--sampled-scores-dir", type=str, default="./shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7", help="Per-meeting .pt files for the SAMPLED importance detector")
    parser.add_argument("--output-tag", type=str, default="", help="Appended to the importance detector name in the output file name, e.g. '_s1' for a second RANDOM seed")
    parser.add_argument("--logp-path", type=str, default=None, help="Override the logp deltas .pt read by the LOGP8B/LOGP32B detectors (e.g. a validation-split file)")

    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.ami_path)
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
    if args.meetings is not None:
        meeting_folders = [f for f in meeting_folders if f.name in set(args.meetings)]

    def get_mis_detector(detector_name, meeting_folder):
        return MIS_CLS_MAP[detector_name](meeting_path=meeting_folder)

    def get_imp_detector(detector_name, meeting_folder, transcript_file):
        extra = {"logp_path": args.logp_path} if args.logp_path is not None else {}
        return IMP_CLS_MAP[detector_name](
            meeting_path=meeting_folder,
            transcript_file=transcript_file,
            sampled_scores_dir=args.sampled_scores_dir,
            **extra,
        )

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
            mis_preds_bool_mask, mis_confs = mis_detector.pred_mistranscribed(line_numbers, return_conf=True)
            num_mistranscribed = sum(mis_preds_bool_mask)
            logger.info(f"Mistranscription detector predicted: {num_mistranscribed} mistranscribed lines out of {num_lines} lines")

            # Retrieve Importance Detector
            imp_detector = get_imp_detector(args.importance_detector, meeting_folder, TRANSCRIPT_FILE)
            imp_preds_bool_mask, imp_confs = imp_detector.get_important_lines(line_numbers, return_conf=True)
            num_important = sum(imp_preds_bool_mask)
            logger.info(f"Importance detector predicted: {num_important} important lines out of {num_lines} lines")

            # Combine confidences
            final_scores = [mis_confs[i] + imp_confs[i] for i in line_numbers]
            
            # Shuffle line_numbers so ties are broken randomly, then rank ALL lines by final_scores
            random.shuffle(line_numbers)
            ranked_lines = sorted(line_numbers, key=lambda i: final_scores[i], reverse=True)
            
            # Select top args.num_lines
            imp_line_idxs = ranked_lines[:args.num_lines]
            logger.info(f"Selected top {len(imp_line_idxs)} lines based on combined confidences.")

            

            # For each idx 
            for chosen_idx in imp_line_idxs:
                chosen_line = original_transcript_lines[chosen_idx]
                updated_transcript_lines[chosen_idx] = gt_lines[chosen_idx]

                # Extract timestamps for the chosen line
                start_time, end_time = extract_timestamps(chosen_line)

                logger.info(f"Clarified timestamps: {start_time} - {end_time}")

            logger.info(f"Total lines clarified: {len(imp_line_idxs)}")

            # ┌───────────────────────────────────────────────┐
            # │                     SAVE                      │
            # └───────────────────────────────────────────────┘
            clarified_file_name = f"{TRANSCRIPT_FILE.split('.')[0]}_{args.mistranscription_detector.lower()}_{args.importance_detector.lower()}{args.output_tag}_{args.num_lines}_clarify4.txt"
            fixed_transcript_file_path = meeting_folder / "transcripts" / clarified_file_name
            with open(fixed_transcript_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(updated_transcript_lines))
                
            logger.info(f"Saved transcript for {fixed_transcript_file_path}\n\n")

