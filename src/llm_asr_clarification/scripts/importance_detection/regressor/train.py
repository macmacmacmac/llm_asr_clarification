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
class ImportanceDataset(Dataset):
    def __init__(self, data_path, use_feats=True):
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        self.contexts = data["context_embs"].float()
        self.targets = data["target_embs"].float()
        self.labels = data["labels"].float()

        # Optional RF mistranscription features of the target line
        self.use_feats = use_feats
        if use_feats:
            if "target_feats" not in data:
                raise ValueError(f"{data_path} has no target_feats; recompile the dataset or pass --no-feats")
            self.target_feats = data["target_feats"].float()

    @property
    def num_feats(self):
        return self.target_feats.shape[1] if self.use_feats else 0

    def feat_stats(self):
        mean = self.target_feats.mean(dim=0)
        std = self.target_feats.std(dim=0).clamp(min=1e-6)
        return mean, std

    def standardize_feats(self, mean, std):
        self.target_feats = (self.target_feats - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "context": self.contexts[idx],
            "target": self.targets[idx],
            "label": self.labels[idx]
        }
        if self.use_feats:
            item["target_feats"] = self.target_feats[idx]
        return item


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


def transform_labels(y, kind):
    # Sign-preserving log1p so negative (v1) labels are handled
    if kind == "log1p":
        return torch.sign(y) * torch.log1p(y.abs())
    return y


def inverse_transform_labels(y, kind):
    if kind == "log1p":
        return torch.sign(y) * torch.expm1(y.abs())
    return y


def forward_batch(model, batch, device):
    context = batch["context"].to(device)
    target = batch["target"].to(device)
    if "target_feats" in batch:
        return model(context, target, batch["target_feats"].to(device))
    return model(context, target)


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
    parser = argparse.ArgumentParser(description="Train ImportanceLSTM regression model")
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/logp_regression_datasets/with_rf_features_v2")
    parser.add_argument("--out-dir", type=str, default="./shared/model_weights/logp_regressor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lstm-layers", type=int, default=1, help="# of LSTM layers to use")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"], help="Training loss")
    parser.add_argument("--label-transform", type=str, default="none", choices=["none", "log1p"],
                        help="Transform applied to labels for training; metrics are reported in original label space")
    parser.add_argument("--no-feats", action="store_true", help="Ignore RF scalar features (embeddings-only ablation)")
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    OUT_DIR = Path(args.out_dir) / DATASET_PATH.name
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr

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
    logger.info(f"Loading training data from {DATASET_PATH}")
    train_dataset = ImportanceDataset(DATASET_PATH / "train.pt", use_feats=not args.no_feats)
    val_dataset = ImportanceDataset(DATASET_PATH / "validation.pt", use_feats=not args.no_feats)

    # Standardize scalar features using train-set statistics
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    num_feats = train_dataset.num_feats
    if num_feats > 0:
        feat_mean, feat_std = train_dataset.feat_stats()
        train_dataset.standardize_feats(feat_mean, feat_std)
        val_dataset.standardize_feats(feat_mean, feat_std)
        feat_stats_path = OUT_DIR / "feat_stats.pt"
        torch.save({"mean": feat_mean, "std": feat_std}, feat_stats_path)
        logger.info(f"Using {num_feats} scalar features; saved standardization stats to {feat_stats_path}")
    else:
        logger.info("Not using scalar features (embeddings only)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Train target stats -> {format_label_stats(train_dataset.labels)}")
    logger.info(f"Val target stats   -> {format_label_stats(val_dataset.labels)}")


    # ┌───────────────────────────────────────────────┐
    # │          INIT MODEL, LOSS, OPTIMIZER          │
    # └───────────────────────────────────────────────┘
    model = ImportanceLSTM(num_layers=args.lstm_layers, num_feats=num_feats).to(DEVICE)
    loss_fn = nn.HuberLoss() if args.loss == "huber" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    best_val_preds = None
    best_val_labels = None
    training_stats = []

    for epoch in range(EPOCHS):

        # ┌───────────────────────────────────────────────┐
        # │                TRAINING LOOP                  │
        # └───────────────────────────────────────────────┘
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            label = transform_labels(batch["label"].to(DEVICE), args.label_transform)

            optimizer.zero_grad()
            preds = forward_batch(model, batch, DEVICE)
            loss = loss_fn(preds, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss ({args.loss}): {avg_train_loss:.4f}")

        # ┌───────────────────────────────────────────────┐
        # │                VALIDATION LOOP                │
        # └───────────────────────────────────────────────┘
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                raw_label = batch["label"].to(DEVICE)
                label = transform_labels(raw_label, args.label_transform)

                preds = forward_batch(model, batch, DEVICE)
                loss = loss_fn(preds, label)
                val_loss += loss.item()

                # Metrics are computed in the original label space
                all_preds.extend(inverse_transform_labels(preds, args.label_transform).cpu().numpy())
                all_labels.extend(raw_label.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        r2, pearson_r, spearman_r = get_metrics(all_labels, all_preds)
        
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} - Val Loss ({args.loss}): {avg_val_loss:.4f} | "
            f"R2: {r2:.4f} | Pearson R: {pearson_r:.4f} | Spearman R: {spearman_r:.4f}"
        )

        training_stats.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "r2": round(r2, 4),
            "pearson_r": round(pearson_r, 4),
            "spearman_r": round(spearman_r, 4)
        })

        # Save Best Model and its predictions
        if val_loader and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_preds = all_preds
            best_val_labels = all_labels
            best_ckpt_path = OUT_DIR / "logp_delta_lstm.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"Validation loss improved to {best_val_loss:.4f}. Saved best model to {best_ckpt_path}")

        logger.info("=" * 50)

    # Save training stats to CSV
    csv_path = OUT_DIR / "training_stats.csv"
    df = pd.DataFrame(training_stats)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved training statistics to {csv_path}")

    # Save validation predictions from best epoch for downstream analysis
    if best_val_preds is not None:
        preds_path = OUT_DIR / "val_predictions.pt"
        torch.save({
            "labels": np.array(best_val_labels),
            "preds": np.array(best_val_preds)
        }, preds_path)
        logger.info(f"Saved best-epoch validation predictions to {preds_path}")

    logger.info("Training complete!")
