import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import random
import re
import numpy as np
import json
import torch
from sentence_transformers.util import cos_sim
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
    ALLImportanceDetector,
    OracleMaskImportanceDetector,
    SIM_MODEL
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
    'ORACLEMASK' : OracleMaskImportanceDetector
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


def diversity_suffix(weight):
    """File name suffix for a diversity weight: '' for 0, else e.g. '_div1' or '_div0.5'."""
    return "" if weight == 0 else f"_div{weight:g}"


def pick_diverse_lines(line_order, scores, embeds, k, weight):
    """
    Greedily pick k lines, one at a time, by rank = score + weight * diversity.

    diversity(line) = 1 - (highest cosine similarity between the line and any line already picked).
    Before anything is picked, diversity is 1 for every line.

    line_order: line indices, already shuffled. Ties go to the line that comes first in this order.
    scores:     combined mistranscription + importance score for every line
    embeds:     sentence embeddings for every line, shape (num_lines, dim)

    Returns the picked line indices in pick order, and a list of
    {"line", "score", "diversity", "rank"} for each pick, as they were when it was picked.
    """
    # Shapes: N = number of lines in the meeting, D = embedding size (384 for MiniLM).
    # Every tensor below is in shuffled order: position p holds line number line_order[p].
    order = torch.tensor(line_order, device=embeds.device)                    # (N,)  line numbers
    # float64 so scores that differ only slightly don't round to the same value and change the order
    order_scores = torch.tensor([scores[i] for i in line_order], dtype=torch.float64, device=embeds.device)  # (N,)
    order_embeds = embeds[order]                                              # (N, D)

    # max_sim[p] = highest similarity between line p and ANY line picked so far.
    # Nothing is picked yet, so all 0 (diversity 1)
    max_sim = torch.zeros(len(line_order), dtype=torch.float64, device=embeds.device)   # (N,)
    available = torch.ones(len(line_order), dtype=torch.bool, device=embeds.device)     # (N,)  False once picked

    picked, picks_info = [], []
    for _ in range(min(k, len(line_order))):
        diversity = 1 - max_sim                                               # (N,)
        rank = order_scores + weight * diversity                              # (N,)
        rank[~available] = float("-inf")                                      # (N,)  picked lines can't win again

        # argmax returns the first maximum, so ties follow the shuffled order
        pos = int(torch.argmax(rank))                                         # scalar: position of the winner
        line = line_order[pos]
        picked.append(line)
        picks_info.append({
            "line": line,
            "score": float(order_scores[pos]),
            "diversity": float(diversity[pos]),
            "rank": float(rank[pos]),
        })
        available[pos] = False

        # Update each line's highest similarity using only the line just picked.
        # order_embeds[pos] is (D,); unsqueeze(0) makes it (1, D) because cos_sim needs 2-D inputs.
        # cos_sim((1, D), (N, D)) gives (1, N): similarity of the new pick to every line.
        # squeeze(0) drops the extra dim, giving (N,).
        sim_to_new = cos_sim(order_embeds[pos].unsqueeze(0), order_embeds).squeeze(0)   # (N,)
        # Element by element: max_sim[p] = larger of (old best for line p, similarity of line p to the new pick).
        # The old best already covers all earlier picks, so the result covers every pick so far.
        max_sim = torch.maximum(max_sim, sim_to_new.double())                 # (N,)

    return picked, picks_info


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
    parser.add_argument("--diversity-weight", type=float, default=0.0, help="w in rank = score + w * diversity. 0 keeps plain top-k ranking")
    parser.add_argument("--rougeL-threshold", type=float, default=None, help="GT mistranscription detector only: flag lines with ROUGE-L below this (default: the detector's built-in 0.5)")
    parser.add_argument("--mis-binary", action="store_true", default=False, help="Rank by the mistranscription detector's yes/no labels instead of its confidence: flagged lines come first, then importance breaks ties")
    parser.add_argument("--mis-name", type=str, default=None, help="Replaces the mistranscription detector name in output file names, e.g. gtbin30")

    args, _ = parser.parse_known_args(args_list)

    # --mis-binary ranks by (flag, importance) pairs, which can't be added to a diversity score
    if args.diversity_weight > 0 and args.mis_binary:
        parser.error("--diversity-weight can't be used with --mis-binary")

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
        if args.rougeL_threshold is not None:
            kwargs["rougeL_threshold"] = args.rougeL_threshold
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
            if args.mis_binary:
                # Every flagged line ranks above every unflagged one; importance orders lines within each group
                final_scores = [(int(mis_preds_bool_mask[i]), imp_confs[i]) for i in line_numbers]
            elif args.score_op == "multiply":
                final_scores = [mis_confs[i] * imp_confs[i] for i in line_numbers]
            else:
                final_scores = [mis_confs[i] + imp_confs[i] for i in line_numbers]
            
            # Shuffle line_numbers so ties are broken randomly
            random.shuffle(line_numbers)

            if args.diversity_weight > 0:
                # Greedy: pick one line at a time by score + w * diversity
                embeds = SIM_MODEL.encode(
                    [line.split(":")[-1] for line in original_transcript_lines], convert_to_tensor=True
                )
                imp_line_idxs, picks_info = pick_diverse_lines(
                    line_numbers, final_scores, embeds, args.num_lines, args.diversity_weight
                )
                logger.info(f"Selected {len(imp_line_idxs)} lines with greedy diversity picking (w={args.diversity_weight}).")
            else:
                # Rank ALL lines by final_scores and select the top args.num_lines
                ranked_lines = sorted(line_numbers, key=lambda i: final_scores[i], reverse=True)
                imp_line_idxs = ranked_lines[:args.num_lines]
                picks_info = [
                    {"line": i, "score": final_scores[i], "diversity": None, "rank": final_scores[i]}
                    for i in imp_line_idxs
                ]
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
            mis_name = args.mis_name or args.mistranscription_detector.lower()
            op_suffix = "_mul" if args.score_op == "multiply" and not args.mis_binary else ""
            div_suffix = diversity_suffix(args.diversity_weight)
            clarified_stem = f"{TRANSCRIPT_FILE.split('.')[0]}_{mis_name}_{importance_name}_{args.num_lines}{op_suffix}{div_suffix}_clarify4"
            output_dir = meeting_folder / "transcripts" / args.output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            fixed_transcript_file_path = output_dir / f"{clarified_stem}.txt"
            with open(fixed_transcript_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(updated_transcript_lines))

            # Save the picked lines, in pick order, for later analysis
            picks_path = output_dir / f"{clarified_stem}_picks.json"
            with open(picks_path, "w", encoding="utf-8") as f:
                json.dump({"diversity_weight": args.diversity_weight, "picks": picks_info}, f, indent=2)

            logger.info(f"Saved transcript for {fixed_transcript_file_path}\n\n")

