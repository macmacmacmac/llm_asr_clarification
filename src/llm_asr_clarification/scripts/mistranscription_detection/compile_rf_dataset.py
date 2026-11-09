"""
Compile the mistranscription detection dataset for the random forest, one file per split.

Each row is one transcript line, built from a meeting's beam results file:
    inputs    the RF features listed in RF_FEATURES (other beams and text are kept for later analysis)
    label     y_true = 1 if ROUGE-L between the top beam and the GT line is below --rouge-threshold

Choose the transcript with --beam-file: beam_results (clean), beam_results_noise or beam_results_partial_noise.
Output: <out-dir>/<beam-file>/<split>.json, a list of records readable with pandas.read_json.

Replaces the data preparation in notebooks/log_prob_beams_elbowplot.ipynb, with the same features, label and
token limit. Lines over the token limit are dropped, so use line_idx rather than row position to find a line.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from jiwer import wer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from llm_asr_clarification import get_logger
from llm_asr_clarification.models.MistranscriptionDetector import (
    JIWER_TRANSFORM,
    LLM_TOKENIZER,
    normalize_text,
    rouge_l,
)

NUM_BEAMS = 5

# The inputs of shared/model_weights/rf_model2.pkl
RF_FEATURES = [
    "num_tokens_text",
    "beam_1_llmlogprob",
    "beam_1_asrlogprob",
    "max_llmlogprobs",
    "min_llmlogprobs",
    "spread_llmlogprobs",
    "max_asrlogprobs",
    "min_asrlogprobs",
    "spread_asrlogprobs",
]


def compute_wer(hypothesis: str, reference: str) -> float:
    """WER capped at 1. None if the GT has no words left after normalization, since WER is undefined there."""
    if not normalize_text(reference):
        return None
    return min(1.0, wer(reference=reference, hypothesis=hypothesis, reference_transform=JIWER_TRANSFORM, hypothesis_transform=JIWER_TRANSFORM))


def build_meeting_rows(beam_results: list[dict], meeting_name: str, rouge_threshold: float) -> pd.DataFrame:
    rows = []
    for line_idx, line in enumerate(beam_results):
        row = {"meeting_name": meeting_name, "line_idx": line_idx, "gt": line["gt"]}
        for i in range(1, NUM_BEAMS + 1):
            beam = line[f"beam_{i}"]
            row[f"beam_{i}_text"] = beam["text"]
            row[f"beam_{i}_asrlogprob"] = beam["asr_avg_log_prob"]
            row[f"beam_{i}_llmlogprob"] = beam["llm_avg_log_prob"]
        rows.append(row)
    df = pd.DataFrame(rows)

    # Aggregated beam features
    for name, suffix in [("llm", "llmlogprob"), ("asr", "asrlogprob")]:
        values = df[[f"beam_{i}_{suffix}" for i in range(1, NUM_BEAMS + 1)]].to_numpy()
        df[f"max_{name}logprobs"] = values.max(axis=1)
        df[f"min_{name}logprobs"] = values.min(axis=1)
        df[f"spread_{name}logprobs"] = df[f"max_{name}logprobs"] - df[f"min_{name}logprobs"]

    # The top beam is the transcript line
    df["text"] = df["beam_1_text"]
    df["num_tokens_text"] = [len(LLM_TOKENIZER.encode(t)) for t in df["text"]]
    df["num_tokens_gt"] = [len(LLM_TOKENIZER.encode(t)) for t in df["gt"]]

    # Label
    df["rougeL"] = [rouge_l(normalize_text(t), normalize_text(g)).fmeasure for t, g in zip(df["text"], df["gt"])]
    df["wer"] = [compute_wer(t, g) for t, g in zip(df["text"], df["gt"])]
    df["y_true"] = (df["rougeL"] < rouge_threshold).astype(int)
    return df


# Driver code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami-path", type=str, default="./shared/datasets/amicorpus")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation"])
    parser.add_argument("--beam-file", type=str, default="beam_results_noise", help="Beam results file name, without .json")
    parser.add_argument("--out-dir", type=str, default="./shared/datasets/mistranscription_detector_datasets")
    parser.add_argument("--rouge-threshold", type=float, default=0.5, help="Lines with ROUGE-L below this are labelled 1")
    parser.add_argument("--max-tokens", type=int, default=400, help="Lines with this many tokens or more are dropped")

    args, _ = parser.parse_known_args(args_list)

    # Init Logger
    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    # Log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    out_dir = Path(args.out_dir) / args.beam_file
    out_dir.mkdir(parents=True, exist_ok=True)

    with logging_redirect_tqdm(loggers=[logger]):
        for split in args.splits:
            split_path = Path(args.ami_path) / split
            meeting_folders = sorted(f for f in split_path.iterdir() if f.is_dir())

            meeting_dfs = []
            for meeting_folder in tqdm(meeting_folders, desc=f"Processing {split}"):
                beam_path = meeting_folder / "artifacts" / f"{args.beam_file}.json"
                if not beam_path.exists():
                    logger.warning(f"Skipping {meeting_folder.name}: missing {beam_path.name}")
                    continue

                with open(beam_path, "r", encoding="utf-8") as f:
                    beam_results = json.load(f)
                meeting_dfs.append(build_meeting_rows(beam_results, meeting_folder.name, args.rouge_threshold))

            if not meeting_dfs:
                logger.error(f"No meetings found for {split}. Skipping this split.")
                continue

            df = pd.concat(meeting_dfs, ignore_index=True)
            num_lines = len(df)
            df = df[df["num_tokens_text"] < args.max_tokens].reset_index(drop=True)

            logger.info(
                f"{split}: {df['meeting_name'].nunique()} meetings | {len(df)} lines | "
                f"Dropped over token limit: {num_lines - len(df)} | Label 1: {int(df['y_true'].sum())} ({df['y_true'].mean():.2%}) | "
                f"Lines without a WER: {int(df['wer'].isna().sum())}"
            )
            if df[RF_FEATURES].isna().to_numpy().any():
                logger.warning(f"{split}: some RF features are missing: {df[RF_FEATURES].isna().sum().to_dict()}")

            out_file = out_dir / f"{split}.json"
            df.to_json(out_file, orient="records")
            logger.info(f"Saved {split} dataset to {out_file}")

    logger.info("All tasks completed successfully!")
