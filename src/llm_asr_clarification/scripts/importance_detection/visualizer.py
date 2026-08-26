import os
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from llm_asr_clarification import get_logger

def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Visualize PR curve and find optimal F1 threshold")
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default="./shared/model_weights/importance_lstm/importance_detector_teacher_datasets/", 
        help="Path to the validation predictions .pt file"
    )
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    VAL_PREDS_PATH = Path(args.out_dir) / "val_predictions.pt"
    OUT_DIR = Path(args.out_dir)

    # Init Logger
    global logger
    logger = get_logger(exp_name)    
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    # Log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    if not VAL_PREDS_PATH.exists():
        logger.error(f"Predictions file not found at: {VAL_PREDS_PATH}")
        return

    # Load predictions
    logger.info(f"Loading predictions from {VAL_PREDS_PATH}")
    data = torch.load(VAL_PREDS_PATH, map_location='cpu', weights_only=False)
    labels = data["labels"]
    probs = data["probs"]

    # Calculate Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    
    # Calculate F1 scores across all thresholds
    # Avoid division by zero
    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    # Find the threshold that maximizes F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    logger.info(f"Generating visualizations...")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: PR Curve
    ax1.plot(recall, precision, label='PR Curve', color='blue')
    ax1.scatter([best_recall], [best_precision], color='red', s=100, label='Optimal F1', zorder=5)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: F1 vs Threshold (Elbow style)
    ax2.plot(thresholds, f1_scores, label='F1 Score', color='green')
    ax2.axvline(x=best_threshold, color='red', linestyle='--', label=f'Optimal Threshold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Threshold')
    ax2.grid(True)
    
    # Write optimal values on the plot
    textstr = '\n'.join((
        f'Optimal Threshold: {best_threshold:.4f}',
        f'Best F1 Score: {best_f1:.4f}',
        f'Precision: {best_precision:.4f}',
        f'Recall: {best_recall:.4f}'
    ))
    
    # Place a text box in lower center/right in axes coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.40, 0.30, textstr, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    ax2.legend()

    # Save plots
    out_plot_path = OUT_DIR / "f1_and_pr_curves.png"
    plt.savefig(out_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {out_plot_path}")
