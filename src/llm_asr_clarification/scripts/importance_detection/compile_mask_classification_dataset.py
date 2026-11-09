"""
Compile a binary importance classification dataset from the masked GT labels
(generate_mask_logp_deltas.py).

Each line gets a score: its highest (answer log-prob drop / per-question std) across the quiz questions.
    score >  --pos-threshold   -> label 1
    score <= --neg-threshold   -> label 0
    anything in between        -> not used as a training target (still appears as context for other lines)

Setting both thresholds to the same value keeps every line, e.g.:
    --pos-threshold 5 --neg-threshold 3   lines scoring in (3, 5] are left out
    --pos-threshold 5 --neg-threshold 5   lines scoring in (3, 5] are labelled 0
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from sentence_transformers import SentenceTransformer
from llm_asr_clarification import get_logger

device = "cuda" if torch.cuda.is_available() else "cpu"
SIM_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def clean_segment(s: str) -> str:
    parts = s.split("]: ", 1)
    if len(parts) > 1:
        return parts[1].strip()
    return s.split(":", 1)[-1].strip()


def compute_line_scores(logps: torch.Tensor, tau: float) -> torch.Tensor:
    # logps: [N+1, Q]; row 0 = full GT transcript, row j+1 = line j masked
    drop = logps[0:1] - logps[1:]
    return (drop / drop.std(dim=0).clamp(min=tau)).max(dim=1).values


# Driver code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/amicorpus/validation")
    parser.add_argument("--labels-path", type=str, default="./shared/datasets/logp_mask_deltas/validation_32b_mask.pt")
    parser.add_argument("--out-dir", type=str, default="./shared/datasets/mask_classification_datasets/pos5_neg3")
    parser.add_argument("--transcript-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--pos-threshold", type=float, default=5.0, help="Lines scoring above this are labelled 1")
    parser.add_argument("--neg-threshold", type=float, default=5.0, help="Lines scoring at or below this are labelled 0")
    parser.add_argument("--tau", type=float, default=0.05, help="Floor (nats) for the per-question std")

    args, _ = parser.parse_known_args(args_list)
    if args.neg_threshold > args.pos_threshold:
        raise ValueError(f"--neg-threshold ({args.neg_threshold}) must not exceed --pos-threshold ({args.pos_threshold})")

    DATASET_PATH = Path(args.dataset_path)
    LABELS_PATH = Path(args.labels_path)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Init Logger
    global logger
    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    # Log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    # ┌───────────────────────────────────────────────┐
    # │                   LOAD DATA                   │
    # └───────────────────────────────────────────────┘
    mask_labels: dict = torch.load(LABELS_PATH, map_location="cpu")
    meeting_folders = sorted(f for f in DATASET_PATH.iterdir() if f.is_dir())

    all_context_embs = []
    all_target_embs = []
    all_contexts = []
    all_targets = []
    all_labels = []
    all_scores = []
    all_meetings = []
    all_line_idxs = []
    num_left_out = 0

    with logging_redirect_tqdm(loggers=[logger]):
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
            if meeting_folder.name not in mask_labels:
                continue

            gen_transcript_path = meeting_folder / "transcripts" / args.transcript_file
            if not gen_transcript_path.exists():
                logger.warning(f"Skipping {meeting_folder.name}: missing {gen_transcript_path.name}")
                continue

            with open(gen_transcript_path, "r", encoding="utf-8") as f:
                gen_lines = f.read().strip().split("\n")

            scores = compute_line_scores(mask_labels[meeting_folder.name]["logps"].double(), args.tau)
            if len(scores) != len(gen_lines):
                logger.warning(f"Skipping {meeting_folder.name}: row mismatch (labels {len(scores)}, transcript {len(gen_lines)})")
                continue

            # Inputs are ASR text, since that is all the model has at inference
            clean_gen = [clean_segment(s) for s in gen_lines]
            gen_embeddings = SIM_MODEL.encode(clean_gen, convert_to_tensor=True).cpu()

            for i in range(args.window_size, len(gen_lines)):
                score = scores[i].item()
                if score > args.pos_threshold:
                    label = 1
                elif score <= args.neg_threshold:
                    label = 0
                else:
                    num_left_out += 1
                    continue

                all_context_embs.append(gen_embeddings[i - args.window_size: i])
                all_target_embs.append(gen_embeddings[i])
                all_contexts.append(gen_lines[i - args.window_size: i])
                all_targets.append(gen_lines[i])
                all_labels.append(label)
                all_scores.append(score)
                all_meetings.append(meeting_folder.name)
                all_line_idxs.append(i)

    if not all_labels:
        logger.error("No data extracted. Exiting.")
        return

    # Stack everything into Torch tensors
    logger.info("Stacking tensors...")
    X_context_embs = torch.stack(all_context_embs)
    X_target_embs = torch.stack(all_target_embs)
    Y_labels = torch.tensor(all_labels, dtype=torch.long)

    num_pos = int(Y_labels.sum())
    logger.info(f"Dataset shape: Contexts: {X_context_embs.shape}, Targets: {X_target_embs.shape}, Labels: {Y_labels.shape}")
    logger.info(
        f"Label 1: {num_pos} ({num_pos / len(Y_labels):.2%}) | Label 0: {len(Y_labels) - num_pos} | "
        f"Left out: {num_left_out} | Meetings: {len(set(all_meetings))}"
    )

    # Save the tensors
    out_file = OUT_DIR / f"{DATASET_PATH.name}.pt"
    torch.save({
        "context_embs": X_context_embs,
        "target_embs": X_target_embs,
        "contexts": all_contexts,
        "targets": all_targets,
        "labels": Y_labels,
        "scores": torch.tensor(all_scores, dtype=torch.float32),
        "meetings": all_meetings,
        "line_idxs": all_line_idxs,
    }, out_file)
    logger.info(f"Successfully saved dataset to {out_file}")
