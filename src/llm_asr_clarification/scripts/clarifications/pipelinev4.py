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
    LSTMImportanceDetector,
    ALLImportanceDetector
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
    'ALL' : ALLImportanceDetector
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
    parser.add_argument("--importance-model-path", type=str, default=None, help="Weights for the LSTM importance detector (default: the detector's built-in path)")
    parser.add_argument("--mistranscription-model-path", type=str, default=None, help="Weights for the RF mistranscription detector (default: the detector's built-in path)")
    parser.add_argument("--beam-file", type=str, default="beam_results", help="Beam results file the mistranscription detector reads, without .json. Must match --clarify-file, e.g. beam_results_noise")
    parser.add_argument("--run-name", type=str, default=None, help="Replaces the importance detector name in output file names, e.g. lstmmask")
    parser.add_argument("--score-op", type=str, choices=["add", "multiply"], default="add", help="How to combine mistranscription and importance confidences")
    parser.add_argument("--output-subdir", type=str, default="", help="Subfolder of transcripts/ for the clarified files")
    
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

    def get_mis_detector(detector_name, meeting_folder):
        kwargs = {"meeting_path": meeting_folder, "beam_file": args.beam_file}
        if args.mistranscription_model_path:
            kwargs["model_path"] = args.mistranscription_model_path
        return MIS_CLS_MAP[detector_name](**kwargs)
        
    def get_imp_detector(detector_name, meeting_folder, transcript_file):
        kwargs = {"meeting_path": meeting_folder, "transcript_file": transcript_file}
        if args.importance_model_path:
            kwargs["model_path"] = args.importance_model_path
        return IMP_CLS_MAP[detector_name](**kwargs)

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
            if args.score_op == "multiply":
                final_scores = [mis_confs[i] * imp_confs[i] for i in line_numbers]
            else:
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
            importance_name = args.run_name or args.importance_detector.lower()
            op_suffix = "_mul" if args.score_op == "multiply" else ""
            clarified_file_name = f"{TRANSCRIPT_FILE.split('.')[0]}_{args.mistranscription_detector.lower()}_{importance_name}_{args.num_lines}{op_suffix}_clarify4.txt"
            output_dir = meeting_folder / "transcripts" / args.output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            fixed_transcript_file_path = output_dir / clarified_file_name
            with open(fixed_transcript_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(updated_transcript_lines))
                
            logger.info(f"Saved transcript for {fixed_transcript_file_path}\n\n")

