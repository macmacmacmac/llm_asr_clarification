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
from torch.nn.utils.rnn import pad_sequence
import ipdb

device = "cuda" if torch.cuda.is_available() else "cpu"
SIM_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def clean_segment(s: str) -> str:
    parts = s.split("]: ", 1)
    if len(parts) > 1:
        return parts[1].strip()
    return s.split(":", 1)[-1].strip()



# Driver code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/amicorpus/validation")
    parser.add_argument("--labels-path", type=str, default="./shared/datasets/gold_labels/validation_Qwen3-8B-FP8_5w_5s_full.pt")
    parser.add_argument("--transcript-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--out-dir", type=str, default="./shared/datasets/gold_datasets")
    parser.add_argument("--window-size", type=int, default=10)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    LABELS_PATH = Path(args.labels_path)
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
    gold_labels: dict = torch.load(LABELS_PATH)
    meeting_folders = [f for f in DATASET_PATH.iterdir() if f.is_dir()]

    # Lists to store contexts, targets and labels
    all_contexts_embs = []
    all_targets_embs = []
    all_contexts = []
    all_targets = []
    all_labels = []

    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):

            gen_transcript_path = meeting_folder / "transcripts" / args.transcript_file
            
            # Skip if either of GT Transcript, Gen Transcript, or Quiz does not exist for this meeting
            if not gen_transcript_path.exists():
                logger.warning(f"Skipping {meeting_folder.name} due to missing files.")
                continue
                
            with open(gen_transcript_path, "r") as f:
                gen_lines = f.read().strip().split("\n")

            # Clean all Gen segments of Speaker early
            clean_gen = [clean_segment(s) for s in gen_lines]


            # Retrieve Labels from Gold Labels Dict if it exists for the given meeting
            if meeting_folder.name not in gold_labels:
                continue
            labels = torch.tensor([int(l) for l in gold_labels[meeting_folder.name]])

            # Generate Embeddings for gen segments
            gen_embeddings = SIM_MODEL.encode(clean_gen, convert_to_tensor=True).cpu()


            # Create sliding windows for each label
            for i in range(args.window_size, len(gen_lines)):
                # Get context: last 10 segments
                context_embs = gen_embeddings[i - args.window_size: i]
                all_contexts.append(gen_lines[i - args.window_size: i])

                # Get target: current segment
                target_embs = gen_embeddings[i]

                # Get label for this segment
                label = labels[i]

                # Append context, target and label into lists
                all_targets.append(gen_lines[i])
                all_contexts_embs.append(context_embs)
                all_targets_embs.append(target_embs)
                all_labels.append(label)


    # If no context was found, exit the script
    if not all_contexts:
        logger.error("No data extracted. Exiting.")
        return

    # Stack all context, targets and labels into Torch tensors
    logger.info("Stacking tensors...")
    X_context_embs = torch.stack(all_contexts_embs)
    X_target_embs = torch.stack(all_targets_embs)
    Y_labels = torch.stack(all_labels)

    # Log the shape of all tensors
    logger.info(f"Dataset shape: Contexts: {X_context_embs.shape}, Targets: {X_target_embs.shape}, Labels: {Y_labels.shape}")

    # Save the tensors
    out_file = OUT_DIR / f"{DATASET_FILE_NAME}.pt"
    os.makedirs(out_file.parent, exist_ok=True)

    dataset_dict = {
        "context_embs": X_context_embs,
        "target_embs": X_target_embs,
        "contexts": all_contexts,
        "targets": all_targets,
        "labels": Y_labels
    }
    
    torch.save(dataset_dict, out_file)
    logger.info(f"Successfully saved dataset to {out_file}")
