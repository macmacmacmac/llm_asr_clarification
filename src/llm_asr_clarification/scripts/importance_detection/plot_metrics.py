import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
from llm_asr_clarification import get_logger

def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV")
    parser.add_argument("--trained-model-path", type=str, default="./shared/model_weights/importance_lstm/importance_detector_student_datasets")
    args = parser.parse_args(args_list)

    # Parse CLI arguments to global variables
    TRAINED_MODEL_PATH = Path(args.trained_model_path)

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

    if not TRAINED_MODEL_PATH.exists():
        print(f"Error: Could not find {TRAINED_MODEL_PATH}")
        return

    # Load epoch-level stats
    df = pd.read_csv(TRAINED_MODEL_PATH / "training_stats.csv")


    # Determine plots layout: 2x2
    _, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax_loss, ax_auc = axes[0]
    ax_roc, ax_pr = axes[1]
    

    # ──────────────────────────────────────────────
    # Plot 1: Train & Validation Loss
    # ──────────────────────────────────────────────
    ax_loss.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', color='blue')
    if df['val_loss'].notna().any():
        ax_loss.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o', color='orange')
    
    ax_loss.set_title('Training & Validation Loss Trend')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    
    # Force integer ticks on x-axis for epochs
    ax_loss.set_xticks(df['epoch'])

    # ──────────────────────────────────────────────
    # Plot 2: ROC-AUC & PR-AUC over Epochs
    # ──────────────────────────────────────────────
    if df['val_loss'].notna().any() and 'roc_auc' in df.columns:
        ax_auc.plot(df['epoch'], df['roc_auc'], label='ROC-AUC', marker='o')
        if 'pr_auc' in df.columns:
            ax_auc.plot(df['epoch'], df['pr_auc'], label='PR-AUC', marker='s')
        
        ax_auc.set_title('Validation AUC Metrics Trend')
        ax_auc.set_xlabel('Epoch')
        ax_auc.set_ylabel('Score')
        ax_auc.set_ylim(-0.05, 1.05) # Metrics are between 0 and 1
        ax_auc.legend()
        ax_auc.grid(True, linestyle='--', alpha=0.7)
        ax_auc.set_xticks(df['epoch'])
    else:
        ax_auc.text(0.5, 0.5, "No Validation Data Available", ha='center', va='center', fontsize=14)
        ax_auc.set_axis_off()

    # ──────────────────────────────────────────────
    # Plots 3 & 4: ROC Curve and Precision-Recall Curve
    # (only if per-sample predictions are available)
    # ──────────────────────────────────────────────
    data = torch.load(TRAINED_MODEL_PATH / "val_predictions.pt", map_location='cpu', weights_only=False)
    labels = np.array(data["labels"])
    probs = np.array(data["probs"])

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc_val = auc(fpr, tpr)

    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
    ax_roc.set_title('ROC Curve (Best Epoch)')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, linestyle='--', alpha=0.7)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    baseline = labels.sum() / len(labels)

    ax_pr.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AP = {ap:.4f})')
    ax_pr.axhline(y=baseline, color='navy', lw=1, linestyle='--', label=f'Random Classifier ({baseline:.4f})')
    ax_pr.set_title('Precision-Recall Curve (Best Epoch)')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.legend(loc='upper right')
    ax_pr.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save the figure
    out_path = TRAINED_MODEL_PATH / "training_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved metrics plot to {out_path}")
