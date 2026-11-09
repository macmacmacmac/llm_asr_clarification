"""
Train the random forest mistranscription detector on a dataset from compile_rf_dataset.py.

Reports ROC-AUC and top-k precision for the train and validation splits. Train numbers are measured on the
lines the forest was fit on, so expect them to look better than validation.

Outputs:
    <out-path>                      the model, usable as RFDetector(model_path=...) and by pipelinev4 through
                                    --mistranscription-model-path
    <out-path>_metrics.json         the reported metrics
    <out-path>_val_predictions.csv  validation predictions, to recompute the metrics independently
"""

import os
import json
import pickle
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from llm_asr_clarification import get_logger
from llm_asr_clarification.metrics import top_k_precision
from llm_asr_clarification.scripts.mistranscription_detection.compile_rf_dataset import RF_FEATURES


def evaluate(df: pd.DataFrame, probs) -> dict:
    return {
        "num_lines": len(df),
        "positive_rate": float(df["y_true"].mean()),
        "roc_auc": float(roc_auc_score(df["y_true"], probs)),
        "top_k_precision": top_k_precision(df["y_true"], probs, df["meeting_name"]),
    }


def format_metrics(split: str, metrics: dict) -> str:
    top_k = metrics["top_k_precision"]
    baseline = next(iter(top_k.values()))["baseline"]
    lines = "\n".join(f"|---> k={k}: {v['mean']:.4f} (std across meetings {v['std']:.4f})" for k, v in top_k.items())
    return (
        f"{split}: {metrics['num_lines']} lines | Label 1: {metrics['positive_rate']:.2%} | ROC-AUC: {metrics['roc_auc']:.4f}\n"
        f"Top-k precision (random picks: {baseline:.4f}):\n{lines}"
    )


# Driver code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./shared/datasets/mistranscription_detector_datasets/beam_results_noise")
    parser.add_argument("--out-path", type=str, default="./shared/model_weights/rf_model_noise.pkl")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--min-samples-leaf", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    args, _ = parser.parse_known_args(args_list)

    # Init Logger
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
    dataset_path = Path(args.dataset_path)
    columns = RF_FEATURES + ["y_true", "meeting_name", "line_idx"]
    train_df = pd.read_json(dataset_path / "train.json")[columns]
    val_df = pd.read_json(dataset_path / "validation.json")[columns]

    # ┌───────────────────────────────────────────────┐
    # │                     TRAIN                     │
    # └───────────────────────────────────────────────┘
    # Passing a DataFrame stores feature_names_in_, which RFDetector reads to select its inputs
    logger.info(f"Training on {len(train_df)} lines with features: {RF_FEATURES}")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(train_df[RF_FEATURES], train_df["y_true"])

    # ┌───────────────────────────────────────────────┐
    # │                   EVALUATE                    │
    # └───────────────────────────────────────────────┘
    val_probs = model.predict_proba(val_df[RF_FEATURES])[:, 1]
    metrics = {
        "train": evaluate(train_df, model.predict_proba(train_df[RF_FEATURES])[:, 1]),
        "validation": evaluate(val_df, val_probs),
    }
    for split, split_metrics in metrics.items():
        logger.info(format_metrics(split, split_metrics))

    # ┌───────────────────────────────────────────────┐
    # │                     SAVE                      │
    # └───────────────────────────────────────────────┘
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {out_path}")

    metrics_path = out_path.with_name(f"{out_path.stem}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")

    predictions_path = out_path.with_name(f"{out_path.stem}_val_predictions.csv")
    val_df[["meeting_name", "line_idx", "y_true"]].assign(prob=val_probs).to_csv(predictions_path, index=False)
    logger.info(f"Saved validation predictions to {predictions_path}")

    logger.info("All tasks completed successfully!")
