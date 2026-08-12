import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run(args_list=None):
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV")
    parser.add_argument("--in-csv", type=str, default="./shared/model_weights/importance_lstm/training_stats.csv", help="Path to training_stats.csv")
    parser.add_argument("--out-img", type=str, default="./shared/model_weights/importance_lstm/training_metrics.png", help="Path to save the output image")
    
    args = parser.parse_args(args_list)

    if not Path(args.in_csv).exists():
        print(f"Error: Could not find {args.in_csv}")
        return

    # Load data
    df = pd.read_csv(args.in_csv)

    # Create a figure with two side-by-side subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Train & Validation Loss
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', color='blue')
    if df['val_loss'].notna().any():
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o', color='orange')
    
    ax1.set_title('Training & Validation Loss Trend')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Force integer ticks on x-axis for epochs
    ax1.set_xticks(df['epoch'])

    # Plot 2: Evaluation Metrics
    if df['val_loss'].notna().any():
        ax2.plot(df['epoch'], df['accuracy'], label='Accuracy', marker='o')
        ax2.plot(df['epoch'], df['precision'], label='Precision', marker='s')
        ax2.plot(df['epoch'], df['recall'], label='Recall', marker='^')
        ax2.plot(df['epoch'], df['f1_score'], label='F1 Score', marker='d')
        
        ax2.set_title('Validation Metrics Trend')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_ylim(-0.05, 1.05) # Metrics are between 0 and 1
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xticks(df['epoch'])
    else:
        ax2.text(0.5, 0.5, "No Validation Data Available", ha='center', va='center', fontsize=14)
        ax2.set_axis_off()

    plt.tight_layout()
    
    # Save the figure
    out_path = Path(args.out_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved metrics plot to {out_path}")
