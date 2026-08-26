import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import ipdb
import random
import re
from llm_asr_clarification.models import MistranscriptionDetector
from llm_asr_clarification.models.MistranscriptionDetector import (
    RandomBernoulliDetector,
    GTDetector,
    RFDetector
)
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import numpy as np

SAMPLING_RATE = 16_000
GROUND_TRUTH_TRANSCRIPT = "parsed_diarized_gt.txt"
SIM_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device = "cuda")

CLS_MAP = {
    'RANDOM' : RandomBernoulliDetector,
    'RF' : RFDetector,
    'GT' : GTDetector
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

# def print_timestamps(idx_pair: Tuple[int], transcript_lines: List[str]):
#     logger.info("Processing indices")
#     for idx in idx_pair:
#         line = transcript_lines[idx]
#         start_time, end_time = extract_timestamps(line)
#         logger.info(f"idx: {idx}, line: {line}, start: {start_time}, end: {end_time}")
#     logger.info("---")


def get_most_relevant_segment(gold_answers, transcript_segments, k = 5) -> list[int]:
    # Clean segments from speaker and time info
    clean_segments = [s.split(":")[-1] for s in transcript_segments]

    # Compute Embeds for cleaned segments and the gold answer
    # Then compute cosine sim matrix
    segments_embeds = SIM_MODEL.encode(clean_segments, convert_to_tensor=True)
    gold_answers_embeds = SIM_MODEL.encode(gold_answers, convert_to_tensor=True)
    cos_matrix = cos_sim(gold_answers_embeds, segments_embeds).detach().cpu() # (gold_answers, segments)

    # Get top k matching segments for each row
    _, gold_idxs = torch.topk(cos_matrix, k=k, dim = 1) # (gold_answers, k)

    # ipdb.set_trace()

    return gold_idxs.view(-1).tolist()


def get_common_segments(pred_segments_idxs, ref_segments_idxs) -> list[int]:
    pred_set =  set(pred_segments_idxs)
    ref_set = set(ref_segments_idxs)
    return list(pred_set.intersection(ref_set))



def get_imp_segment_idxs(pred_segments, pred_segment_idxs, gold_answers, num_imp) -> list[int]:
    # Segments of Speaker and time segment info
    clean_pred_segments = [s.split(":")[-1] for s in pred_segments]

    # Compute Embeds for cleaned segments and gold answers
    # Then compute cosine sim matrix
    segments_embeds = SIM_MODEL.encode(clean_pred_segments, convert_to_tensor=True)
    gold_answers_embeds = SIM_MODEL.encode(gold_answers, convert_to_tensor=True)
    cos_matrix = cos_sim(gold_answers_embeds, segments_embeds)

    # Normalize along dim 1 (column values)
    mean = cos_matrix.mean(dim=1, keepdim=True)
    std = cos_matrix.std(dim=1, keepdim=True)
    norm_cos_matrix = (cos_matrix - mean) / (std + 1e-8)

    # Sum along all rows (dim 0)
    cos_scores = norm_cos_matrix.sum(dim = 0)
    _, indices = torch.topk(cos_scores, k = num_imp)

    # ipdb.set_trace()

    # Retrieve global indices from local indices and cast to list
    return list(pred_segment_idxs[indices])



# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--transcript-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--detectors", nargs="+", default=["RANDOM", "RF", "GT"])
    parser.add_argument("--num-lines", type=int, default=20)
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

    def get_detector(detector_name, meeting_folder) -> MistranscriptionDetector:
        # if detector_name == "ALL_BAD":
        #     return RandomBernoulliDetector(p = 1, meeting_path = meeting_folder)
        return CLS_MAP[detector_name](meeting_path=meeting_folder)
        
    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
            logger.info(f"Processing meeting {meeting_folder.name}")

            # Load the generated transcript
            transcript_path = meeting_folder / "transcripts" / TRANSCRIPT_FILE
            with open(transcript_path, "r") as f:
                transcript_content = f.read().strip()

            global original_transcript_lines

            # Maintain two lists, one stores original line and one stores updated lines
            original_transcript_lines = transcript_content.split("\n")
            updated_transcript_lines = original_transcript_lines.copy()
            num_lines = len(original_transcript_lines)

            # Load the GT transcript
            gt_transcript_path = meeting_folder / "transcripts" / GT_FILE
            with open(gt_transcript_path, "r") as f:
                gt_content = f.read().strip()
            gt_lines = gt_content.split("\n")


            # Load the quiz for the meeting
            quiz_path = meeting_folder / "quiz" / "quiz_from_parsed_diarized_gt.json"
            with open(quiz_path, "r") as f:
                quiz = json.load(f)

            # Extract gold answers
            gold_answers = [q["correct_answer"] for q in quiz]

            # Get relevant segments for gold answers
            # gold_segments = []
            # gold_idxs = []
            # for a in gold_answers:
            #     relevant_segments, relevant_idxs = get_most_relevant_segment(a, gt_lines)
            #     gold_segments.extend(relevant_segments)
            #     gold_idxs.extend(relevant_idxs)

            gold_idxs = get_most_relevant_segment(gold_answers, gt_lines)
        
            # ipdb.set_trace()

            for detector_name in args.detectors:
                # MISTRANSCRIPTION + IMPORTANCE DETECTION
                # retrieve detector and pred mistranscriptions
                detector = get_detector(detector_name, meeting_folder)
                preds_bool_mask = detector.pred_mistranscribed(range(num_lines))
                logger.info(f"This detector predicted: {sum(preds_bool_mask)} mistranscribed lines out of {num_lines} lines")

                mistranscribed_lines_idxs = np.arange(num_lines)[preds_bool_mask]

                ### ==========IMPORTANCE DETECTION VIA GROUND-TRUTH AND MISTRANSCRIPTION DETECTION METHOD
                imp_line_idxs = get_common_segments(mistranscribed_lines_idxs, gold_idxs)

                ### ==========ONLY IMPORTANCE DETECTION UPPERBOUND NO MISTRANSCRIPTION DETECTION METHOD
                # imp_line_idxs = gold_idxs

                ### ==========RANDOM BASELINE CHOOSING AMONGST MISTRANSCRIPTION DETECTION W/O IMPORTANCE METHOD
                # Select 20 random lines out of the mistranscribed ones
                # imp_line_idxs = np.random.choice(
                #     mistranscribed_lines_idxs, 
                #     min(len(mistranscribed_lines_idxs), args.num_lines), 
                #     replace=False
                # )

                ### LOGGING FOR INSPECTING IMP vs NON-IMP segments
                # logger.info("Important Mistranscribed Segments (& their GT counterparts)")
                # for idx in imp_line_idxs:
                #     logger.info(f"Gen: {original_transcript_lines[idx]}")
                #     logger.info(f"GT: {gt_lines[idx]}\n\n")

                # logger.info("Non-Important Mistranscribed Segments (& their GT counterparts)")
                # for idx in mistranscribed_lines_idxs:
                #     if idx not in imp_line_idxs:
                #         logger.info(f"Gen: {original_transcript_lines[idx]}")
                #         logger.info(f"GT: {gt_lines[idx]}\n\n")


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
                clarified_file_name = TRANSCRIPT_FILE.split(".")[0] + f"_{detector_name.lower()}_clarify2.txt"
                # clarified_file_name = TRANSCRIPT_FILE.split(".")[0] + "_clarify_only_importance.txt"
                fixed_transcript_file_path = meeting_folder / "transcripts" / clarified_file_name
                with open(fixed_transcript_file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(updated_transcript_lines))
                    
                logger.info(f"Saved transcript for {fixed_transcript_file_path}\n\n")











            