"""
Merge per-shard logp-delta files written by generate_logp_deltas.py (--num-shards > 1) into one
{meeting: {...}} file, the format LogpImportanceDetector and the analysis notebooks expect.
Refuses to overwrite a merged file that already has meetings missing from the shards.
"""

import os
import glob
import argparse

import torch

from llm_asr_clarification import get_logger


def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dir", type=str, default="./shared/datasets/logp_deltas/full")
    parser.add_argument("--model-name", type=str, default="Qwen3-32B-FP8")

    args, _ = parser.parse_known_args(args_list)

    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    shard_paths = sorted(glob.glob(os.path.join(args.dir, f"{args.split}_{args.model_name}_shard*of*.pt")))
    if not shard_paths:
        logger.error(f"No shard files found in {args.dir} for {args.split}_{args.model_name}")
        return

    merged = {}
    for path in shard_paths:
        shard = torch.load(path)
        overlap = set(shard) & set(merged)
        if overlap:
            logger.warning(f"{path} repeats meetings {sorted(overlap)}; keeping the first copy")
        merged.update({k: v for k, v in shard.items() if k not in merged})
        logger.info(f"{path}: {len(shard)} meetings")

    out_path = os.path.join(args.dir, f"{args.split}_{args.model_name}.pt")
    if os.path.exists(out_path):
        missing = set(torch.load(out_path)) - set(merged)
        if missing:
            logger.error(f"{out_path} has meetings not in the shards ({sorted(missing)}); not overwriting")
            return
    torch.save(merged, out_path)
    logger.info(f"Merged {len(merged)} meetings from {len(shard_paths)} shards into {out_path}")
