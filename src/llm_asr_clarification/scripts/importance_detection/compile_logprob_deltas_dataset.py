import os
import argparse
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from sentence_transformers import SentenceTransformer
from llm_asr_clarification import get_logger
from llm_asr_clarification.models.MistranscriptionDetector import RFDetector
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
    parser.add_argument("--labels-path", type=str, default="./shared/datasets/logp_deltas/validation_32b_v2.pt")
    parser.add_argument("--transcript-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--out-dir", type=str, default="./shared/datasets/logp_regression_datasets/with_rf_features_v2")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--label-key", type=str, default="segment_deltas", help="Key in labels file, e.g. segment_deltas or segment_deltas_v2")
    
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
    logp_deltas: dict = torch.load(LABELS_PATH)
    meeting_folders = [f for f in DATASET_PATH.iterdir() if f.is_dir()]

    # Lists to store contexts, targets and labels
    all_contexts_embs = []
    all_targets_feats = []
    all_targets_embs = []
    all_contexts = []
    all_targets = []
    all_labels = []

    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):

            # Skip if meeting is not present in logp_deltas
            if meeting_folder.name not in logp_deltas:
                continue

            gen_transcript_path = meeting_folder / "transcripts" / args.transcript_file
            
            # Skip if Gen Transcript does not exist for this meeting 
            if not gen_transcript_path.exists():
                logger.warning(f"Skipping {meeting_folder.name} due to missing files.")
                continue
                
            with open(gen_transcript_path, "r") as f:
                gen_lines = f.read().strip().split("\n")

            # Clean all Gen segments of Speaker Info and timestamp info early
            clean_gen = [clean_segment(s) for s in gen_lines]


            # labels = torch.tensor([int(l) for l in gold_labels[meeting_folder.name]])
            labels = logp_deltas[meeting_folder.name][args.label_key]

            # Per-line RF mistranscription features (row-aligned with beam_results_2.json)
            det = RFDetector(meeting_path=str(meeting_folder))
            if not (len(det.df) == len(gen_lines) == len(labels)):
                logger.warning(
                    f"Skipping {meeting_folder.name}: row mismatch (beam_results {len(det.df)}, "
                    f"transcript {len(gen_lines)}, labels {len(labels)})"
                )
                continue
            feat_cols = list(det.features)
            if "num_tokens_text" not in feat_cols:
                feat_cols.append("num_tokens_text")
            gen_feats = torch.tensor(det.df[feat_cols].values, dtype=torch.float32)  # [N, F]

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
                all_targets_feats.append(gen_feats[i])
                all_labels.append(label)


    # If no context was found, exit the script
    if not all_contexts:
        logger.error("No data extracted. Exiting.")
        return

    # Stack all context, targets and labels into Torch tensors
    logger.info("Stacking tensors...")
    X_context_embs = torch.stack(all_contexts_embs)
    X_target_embs = torch.stack(all_targets_embs)
    X_target_feats = torch.stack(all_targets_feats)
    Y_labels = torch.stack(all_labels)

    # Log the shape of all tensors
    logger.info(f"Dataset shape: Contexts: {X_context_embs.shape}, Targets: {X_target_embs.shape}, Target feats: {X_target_feats.shape}, Labels: {Y_labels.shape}")

    # Save the tensors
    out_file = OUT_DIR / f"{DATASET_FILE_NAME}.pt"
    os.makedirs(out_file.parent, exist_ok=True)

    dataset_dict = {
        "context_embs": X_context_embs,
        "target_embs": X_target_embs,
        "target_feats": X_target_feats,
        "contexts": all_contexts,
        "targets": all_targets,
        "labels": Y_labels
    }
    
    torch.save(dataset_dict, out_file)
    logger.info(f"Successfully saved dataset to {out_file}")
