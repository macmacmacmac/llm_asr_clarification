import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from llm_asr_clarification import get_logger


def run(args_list=None):
    exp_name = os.path.basename(__file__)

    parser = argparse.ArgumentParser(description="Generate scatter plot of predictions vs labels")
    parser.add_argument("--predictions-path", type=str, default="./shared/model_weights/logp_regressor/deterministic/val_predictions.pt")
    args, _ = parser.parse_known_args(args_list)

    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")
    logger.info(f"Predictions path: {args.predictions_path}")

    preds_path = Path(args.predictions_path)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found at: {preds_path}")

    # Load predictions and labels
    data = torch.load(preds_path, map_location="cpu", weights_only=False)
    y_true = data["labels"]
    y_pred = data["preds"]

    # Generate scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_true, alpha=0.3, s=15)
    plt.xlabel("Predicted Delta (y_pred)")
    plt.ylabel("True Delta (y_true)")
    plt.title("Predicted vs True Segment Deltas")
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = preds_path.parent / "scatter.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved scatter plot to {out_path}")
