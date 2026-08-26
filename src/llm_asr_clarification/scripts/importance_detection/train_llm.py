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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from llm_asr_clarification import get_logger
from llm_asr_clarification.scripts.importance_detection import visualizer

# ┌───────────────────────────────────────────────┐
# │               DATASET DEFINITION              │
# └───────────────────────────────────────────────┘
class ImportanceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        data = torch.load(data_path, map_location='cpu')
        self.contexts = data["contexts"]
        self.targets = data["targets"]
        self.labels = data["labels"]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Join context segments if they are provided as a list of strings
        context_str = "\n".join(self.contexts[idx]) if isinstance(self.contexts[idx], list) else self.contexts[idx]
        target_str = self.targets[idx]
        
        text = f"# Context: {context_str}\n# Target: {target_str}"
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
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
    parser = argparse.ArgumentParser(description="Train Importance LLM model with PEFT")
    parser.add_argument("--train-data-path", type=str, default="./shared/datasets/importance_detector_teacher_datasets/train.pt", help="Path to training .pt file")
    parser.add_argument("--val-data-path", type=str, default="./shared/datasets/importance_detector_teacher_datasets/validation.pt", help="Path to validation .pt file")
    parser.add_argument("--out-dir", type=str, default="./shared/model_weights/importance_llm_peft/importance_detector_teacher_datasets", help="Directory to save model weights")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="HF model name")
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    TRAIN_DATA_PATH = args.train_data_path
    VAL_DATA_PATH = args.val_data_path
    OUT_DIR = Path(args.out_dir)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    MODEL_NAME = args.model_name

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
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading training data from {TRAIN_DATA_PATH}")
    train_dataset = ImportanceDataset(TRAIN_DATA_PATH, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Calculate pos_weight for imbalanced dataset
    num_pos = torch.sum(train_dataset.labels == 1).item()
    num_neg = torch.sum(train_dataset.labels == 0).item()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(DEVICE)
    logger.info(f"Dataset Imbalance -> Negatives: {num_neg}, Positives: {num_pos}")
    logger.info(f"Applying pos_weight of {pos_weight.item():.2f} to BCEWithLogitsLoss")


    logger.info(f"Loading validation data from {VAL_DATA_PATH}")
    val_dataset = ImportanceDataset(VAL_DATA_PATH, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    # ┌───────────────────────────────────────────────┐
    # │          INIT MODEL, LOSS, OPTIMIZER          │
    # └───────────────────────────────────────────────┘
    logger.info(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=1, 
        dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(base_model, peft_config).to(DEVICE)
    model.print_trainable_parameters()
    
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

            # Unpack inputs and Label for a mini batch
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            label = batch["label"].float().to(DEVICE)

            # Set Gradients to 0
            optimizer.zero_grad()

            # Forward Pass thru the model
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)

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

                # Unpack inputs and Label for a mini batch
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                label = batch["label"].float().to(DEVICE)

                # Forward Pass thru the model
                # Calculate and Accumulate loss
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(-1)
                    
                    loss = loss_fn(logits, label)
                val_loss += loss.item()

                # Calculate continuous probabilities for ROC-AUC
                probs = torch.sigmoid(logits).float().cpu().numpy()

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
            best_ckpt_path = OUT_DIR
            model.save_pretrained(best_ckpt_path)
            tokenizer.save_pretrained(best_ckpt_path)
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

        visualizer.run(args_list=args_list)

    logger.info("Training complete!")
