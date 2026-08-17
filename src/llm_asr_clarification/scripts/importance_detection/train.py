import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from llm_asr_clarification.models.ImportanceLSTM import ImportanceLSTM
from llm_asr_clarification import get_logger
import ipdb

# ┌───────────────────────────────────────────────┐
# │               DATASET DEFINITION              │
# └───────────────────────────────────────────────┘
class ImportanceDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, map_location='cpu')
        self.contexts = data["context_embs"]
        self.targets = data["target_embs"]
        self.labels = data["labels"]

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
def get_metrics(y_true, y_score):
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    return roc_auc, pr_auc


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Train ImportanceLSTM model")
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/importance_detector_student_datasets", help="Path to training .pt file")
    parser.add_argument("--out-dir", type=str, default="./shared/model_weights/importance_lstm", help="Directory to save model weights")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    
    args = parser.parse_args(args_list)

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
    train_dataset = ImportanceDataset(DATASET_PATH / "train.pt")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = ImportanceDataset(DATASET_PATH / "validation.pt")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate pos_weight for imbalanced dataset
    num_pos = torch.sum(train_dataset.labels == 1).item()
    num_neg = torch.sum(train_dataset.labels == 0).item()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(DEVICE)
    logger.info(f"Dataset Imbalance -> Negatives: {num_neg}, Positives: {num_pos}")
    logger.info(f"Applying pos_weight of {pos_weight.item():.2f} to BCEWithLogitsLoss")


    # ┌───────────────────────────────────────────────┐
    # │          INIT MODEL, LOSS, OPTIMIZER          │
    # └───────────────────────────────────────────────┘
    # Initialize model, loss, optimizer
    model = ImportanceLSTM(num_layers=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create output dir for saving weights, if applicable
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_val_probs = None
    best_val_labels = None
    training_stats = []

    
    for epoch in range(EPOCHS):

        # ┌───────────────────────────────────────────────┐
        # │                TRAINING LOOP                  │
        # └───────────────────────────────────────────────┘

        # Set the model in training mode
        # Init loss as 0
        model.train()
        train_loss = 0.0

        # For all mini batches
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):

            # Unpack Context, Target, and Label for a mini batch
            context = batch["context"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            label = batch["label"].float().to(DEVICE)

            # Set Gradients to 0
            optimizer.zero_grad()

            # Forward Pass thru the model
            logits = model(context, target)

            # Compute Loss 
            loss = loss_fn(logits, label)

            # Calculate gradients and before back propagation
            loss.backward()
            optimizer.step()

            # Accumulate Train Loss
            train_loss += loss.item()

        # Calculate Avg Train Loss
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}")

        # ┌───────────────────────────────────────────────┐
        # │                VALIDATION LOOP                │
        # └───────────────────────────────────────────────┘
        
        # Set the Model in Eval mode
        # Init 0 val loss and empty preds and labels lists for accuracy calculations
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []

        # Freeze the model
        with torch.no_grad():

            # For each mini batch
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):

                # Unpack Context, Target, and Label for a mini batch
                context = batch["context"].to(DEVICE)
                target = batch["target"].to(DEVICE)
                label = batch["label"].float().to(DEVICE)

                # Forward Pass thru the model
                # Calculate and Accumulate loss
                logits = model(context, target)
                loss = loss_fn(logits, label)
                val_loss += loss.item()

                # Calculate continuous probabilities for ROC-AUC
                probs = torch.sigmoid(logits).cpu().numpy()

                # Accumulate Preds and Labels
                all_probs.extend(probs)
                all_labels.extend(label.cpu().numpy())

        # Calculate avg val loss
        avg_val_loss = val_loss / len(val_loader)

        # Calculate eval metrics
        roc_auc, pr_auc = get_metrics(all_labels, all_probs)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {avg_val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

        training_stats.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        })
        

        # Save Best Model and its predictions
        if val_loader and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_probs = all_probs
            best_val_labels = all_labels
            best_ckpt_path = OUT_DIR / "importance_lstm.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"Validation loss improved to {best_val_loss:.4f}. Saved best model to {best_ckpt_path}")

        logger.info("=" * 50)

    # Save training stats to CSV
    csv_path = OUT_DIR / "training_stats.csv"
    df = pd.DataFrame(training_stats)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved training statistics to {csv_path}")

    # Save validation predictions from best epoch for plotting ROC/PR curves
    if best_val_probs is not None:
        preds_path = OUT_DIR / "val_predictions.pt"
        torch.save({
            "labels": np.array(best_val_labels),
            "probs": np.array(best_val_probs)
        }, preds_path)
        logger.info(f"Saved best-epoch validation predictions to {preds_path}")

    logger.info("Training complete!")
