import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from llm_asr_clarification import get_logger

device = "cuda" if torch.cuda.is_available() else "cpu"
SIM_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def clean_segment(s: str) -> str:
    parts = s.split("]: ", 1)
    if len(parts) > 1:
        return parts[1].strip()
    return s.split(":", 1)[-1].strip()


def get_most_relevant_segment(gold_answers, clean_segments, k=5) -> list[int]:
    # Compute Embeds for pre-cleaned segments and the gold answer
    segments_embeds = SIM_MODEL.encode(clean_segments, convert_to_tensor=True)
    gold_answers_embeds = SIM_MODEL.encode(gold_answers, convert_to_tensor=True)
    cos_matrix = cos_sim(gold_answers_embeds, segments_embeds).detach().cpu() # (gold_answers, segments)

    # Get top k matching segments for each row
    _, gold_idxs = torch.topk(cos_matrix, k=k, dim=1) # (gold_answers, k)

    return gold_idxs.view(-1).tolist()


# Driver code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--transcript-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--gt-file", type=str, default="parsed_diarized_gt.txt")
    parser.add_argument("--out-dir", type=str, default="./shared/datasets/importance_detector_datasets")
    parser.add_argument("--window-size", type=int, default=10)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_FILE_NAME = args.dataset_path.split("/")[-1]

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

    # Lists to store contexts, targets and labels
    all_contexts = []
    all_targets = []
    all_labels = []

    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
            gt_transcript_path = meeting_folder / "transcripts" / args.gt_file
            gen_transcript_path = meeting_folder / "transcripts" / args.transcript_file
            quiz_path = meeting_folder / "quiz" / "quiz_from_parsed_diarized_gt.json"

            # Skip if either of GT Transcript, Gen Transcript, or Quiz does not exist for this meeting
            if not gt_transcript_path.exists() or not gen_transcript_path.exists() or not quiz_path.exists():
                logger.warning(f"Skipping {meeting_folder.name} due to missing files.")
                continue

            # Load GT, Gen Transcripts
            with open(gt_transcript_path, "r") as f:
                gt_lines = f.read().strip().split("\n")
                
            with open(gen_transcript_path, "r") as f:
                gen_lines = f.read().strip().split("\n")

            # Load the Quiz
            with open(quiz_path, "r") as f:
                quiz = json.load(f)

            # If Segments of GT and Gen Transcripts differ, then we log and skip
            if len(gt_lines) != len(gen_lines):
                logger.warning(f"Skipping {meeting_folder.name}: GT and Gen transcript lengths differ ({len(gt_lines)} vs {len(gen_lines)}).")
                continue

            # Clean all GT and Gen segments of Speaker early
            clean_gt = [clean_segment(s) for s in gt_lines]
            clean_gen = [clean_segment(s) for s in gen_lines]

            # Retrieve gold answers and their top 5 relevant segments
            gold_answers = [q["correct_answer"] for q in quiz]
            gold_idxs = get_most_relevant_segment(gold_answers, clean_gt)
            # gold_idxs_set = set(gold_idxs)

            # Create Labels
            # Label 0 as default and 1 for all idx == gold idx
            labels = torch.zeros(len(gt_lines), dtype=torch.long)
            labels[gold_idxs] = 1

            # Generate Embeddings for gt and gen segments
            gt_embeddings = SIM_MODEL.encode(clean_gt, convert_to_tensor=True).cpu()
            gen_embeddings = SIM_MODEL.encode(clean_gen, convert_to_tensor=True).cpu()

            
            # Create sliding windows for each label
            for i in range(args.window_size, len(gt_lines)):
                # Get context: last 10 segments
                context = gt_embeddings[i - args.window_size: i]

                # Get target: current segment
                target = gen_embeddings[i]

                # Get label for this segment
                label = labels[i]

                # Append context, target and label into lists
                all_contexts.append(context)
                all_targets.append(target)
                all_labels.append(label)


    # If no context was found, exit the script
    if not all_contexts:
        logger.error("No data extracted. Exiting.")
        return

    # Stack all context, targets and labels into Torch tensors
    logger.info("Stacking tensors...")
    X_context = torch.stack(all_contexts)
    X_target = torch.stack(all_targets)
    Y_labels = torch.stack(all_labels)


    # Log the shape of all tensors
    logger.info(f"Dataset shape: Contexts: {X_context.shape}, Targets: {X_target.shape}, Labels: {Y_labels.shape}")

    # Save the tensors
    out_file = OUT_DIR / f"{DATASET_FILE_NAME}.pt"
    dataset_dict = {
        "contexts": X_context,
        "targets": X_target,
        "labels": Y_labels
    }
    
    torch.save(dataset_dict, out_file)
    logger.info(f"Successfully saved dataset to {out_file}")
