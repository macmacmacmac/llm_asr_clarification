import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from llm_asr_clarification.models.ImportanceLSTM import ImportanceLSTM
from llm_asr_clarification import get_logger


# ┌───────────────────────────────────────────────┐
# │               DATASET DEFINITION              │
# └───────────────────────────────────────────────┘
class FilteredDataset(Dataset):
    def __init__(self, data_path, max_label=3.0):
        data = torch.load(data_path, map_location="cpu", weights_only=False)

        # Filter out all samples with label > max_label
        labels = data["labels"].float()
        keep_mask = labels <= max_label
        self.num_total = len(labels)

        self.contexts = data["context_embs"].float()[keep_mask]
        self.targets = data["target_embs"].float()[keep_mask]
        self.labels = labels[keep_mask]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "context": self.contexts[idx],
            "target": self.targets[idx],
            "label": self.labels[idx]
        }

# ┌───────────────────────────────────────────────┐
# │                 HELPER METHODS                │
# └───────────────────────────────────────────────┘
def get_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(pearsonr(y_true, y_pred).statistic)
        spearman_r = float(spearmanr(y_true, y_pred).statistic)
    else:
        pearson_r = 0.0
        spearman_r = 0.0
    return r2, pearson_r, spearman_r


def format_label_stats(labels: torch.Tensor) -> str:
    arr = labels.numpy()
    return (
        f"n={len(arr)}, mean={arr.mean():.4f}, std={arr.std():.4f}, "
        f"min={arr.min():.4f}, median={np.median(arr):.4f}, "
        f"p99={np.percentile(arr, 99):.4f}, max={arr.max():.4f}"
    )


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/logp_regression_datasets/non-deterministic")
    parser.add_argument("--trained-model-dir", type=str, default="./shared/model_weights/logp_regressor/non-deterministic")
    parser.add_argument("--max-label", type=float, default=3.0, help="Samples with label > this are filtered out")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRAINED_MODEL_DIR = Path(args.trained_model_dir)
    MAX_LABEL = args.max_label
    BATCH_SIZE = args.batch_size

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

    # Determine device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")


    # ┌───────────────────────────────────────────────┐
    # │                   LOAD DATA                   │
    # └───────────────────────────────────────────────┘
    logger.info(f"Loading data from {DATASET_PATH}")

    val_dataset = FilteredDataset(DATASET_PATH / "validation.pt", max_label=MAX_LABEL)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_kept = len(val_dataset)
    num_removed = val_dataset.num_total - num_kept
    logger.info(
        f"Filtered validation set (label <= {MAX_LABEL}): kept {num_kept}/{val_dataset.num_total} "
        f"samples, removed {num_removed} ({100 * num_removed / val_dataset.num_total:.2f}%)"
    )
    logger.info(f"Filtered val target stats -> {format_label_stats(val_dataset.labels)}")


    # ┌───────────────────────────────────────────────┐
    # │               LOAD TRAINED MODEL              │
    # └───────────────────────────────────────────────┘
    ckpt_path = TRAINED_MODEL_DIR / "logp_delta_lstm.pt"
    logger.info(f"Loading trained model from {ckpt_path}")

    model = ImportanceLSTM(num_layers=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()


    # ┌───────────────────────────────────────────────┐
    # │                VALIDATION LOOP                │
    # └───────────────────────────────────────────────┘
    loss_fn = nn.MSELoss()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Filtered Val]"):
            context = batch["context"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            preds = model(context, target)
            val_loss += loss_fn(preds, label).item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    r2, pearson_r, spearman_r = get_metrics(all_labels, all_preds)

    logger.info(
        f"Filtered Val (label <= {MAX_LABEL}) - Val Loss (MSE): {avg_val_loss:.4f} | "
        f"R2: {r2:.4f} | Pearson R: {pearson_r:.4f} | Spearman R: {spearman_r:.4f}"
    )
    logger.info("Validation complete!")
