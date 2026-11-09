"""
Build "oracle" clarified transcripts: pick lines straight from the saved labels instead of a model.

These labels are computed with the quiz answers, so this is not a deployable method. Its quiz score
shows the best that a model trained on the label could hope for.

    oracleswap:    biggest answer log-prob gain from correcting the ASR line (raw, best question per line)
    oracleswapstd: the same gain, std-scaled per question so no single question dominates the picks
    oraclemask:    biggest answer log-prob drop from hiding the GT line (std-scaled like the training label)

The file names follow the pipelinev4 format, with ALL as the mistranscription detector name.
"""

import os
import random
import argparse
from pathlib import Path
import torch
from llm_asr_clarification import get_logger


def swap_scores(logps: torch.Tensor, tau: float = None) -> torch.Tensor:
    # logps: [N+1, Q]; row 0 = full ASR transcript, row j+1 = ASR line j replaced by GT
    # With tau, each question's gain is divided by its std (floored at tau)
    gain = logps[1:] - logps[0:1]
    if tau is not None:
        gain = gain / gain.std(dim=0).clamp(min=tau)
    return gain.max(dim=1).values


def mask_scores(logps: torch.Tensor, tau: float) -> torch.Tensor:
    # logps: [N+1, Q]; row 0 = full GT transcript, row j+1 = GT line j masked
    drop = logps[0:1] - logps[1:]
    return (drop / drop.std(dim=0).clamp(min=tau)).max(dim=1).values


def read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip().split("\n")


# Driver code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami_path", type=str, default="./shared/datasets/amicorpus/validation")
    parser.add_argument("--swap-labels-path", type=str, default="./shared/datasets/logp_deltas/validation_32b.pt")
    parser.add_argument("--mask-labels-path", type=str, default="./shared/datasets/logp_mask_deltas/validation_32b_mask.pt")
    parser.add_argument("--clarify-file", type=str, default="custom_transcript_gt_segments.txt")
    parser.add_argument("--output-subdir", type=str, default="oracle-labels")
    parser.add_argument("--num-lines", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--tau", type=float, default=0.05, help="Floor (nats) for the per-question std in the scaled scores")
    parser.add_argument("--seed", type=int, default=47)

    args, _ = parser.parse_known_args(args_list)

    # Init Logger
    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    # Log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    random.seed(args.seed)

    # ┌───────────────────────────────────────────────┐
    # │                   LOAD DATA                   │
    # └───────────────────────────────────────────────┘
    swap_labels: dict = torch.load(args.swap_labels_path, map_location="cpu")
    mask_labels: dict = torch.load(args.mask_labels_path, map_location="cpu")
    meeting_folders = sorted(f for f in Path(args.ami_path).iterdir() if f.is_dir())
    base_name = args.clarify_file.split(".")[0]

    for meeting_folder in meeting_folders:
        meeting = meeting_folder.name
        transcripts = meeting_folder / "transcripts"
        asr_lines = read_lines(transcripts / args.clarify_file)
        gt_lines = read_lines(transcripts / "parsed_diarized_gt.txt")

        scores_by_name = {
            "oracleswap": swap_scores(swap_labels[meeting]["logps"].double()),
            "oracleswapstd": swap_scores(swap_labels[meeting]["logps"].double(), args.tau),
            "oraclemask": mask_scores(mask_labels[meeting]["logps"].double(), args.tau),
        }

        # Line numbers must match across the transcripts and both label files
        sizes = {len(asr_lines), len(gt_lines), *(len(s) for s in scores_by_name.values())}
        if len(sizes) != 1:
            raise ValueError(f"{meeting}: row mismatch across transcripts and labels: {sizes}")

        output_dir = transcripts / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, scores in scores_by_name.items():
            # Shuffle first so ties are broken randomly, same as pipelinev4
            line_numbers = list(range(len(asr_lines)))
            random.shuffle(line_numbers)
            ranked_lines = sorted(line_numbers, key=lambda i: scores[i].item(), reverse=True)

            for num_lines in args.num_lines:
                updated_lines = asr_lines.copy()
                for idx in ranked_lines[:num_lines]:
                    updated_lines[idx] = gt_lines[idx]

                out_path = output_dir / f"{base_name}_all_{name}_{num_lines}_clarify4.txt"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(updated_lines))

        logger.info(f"Saved transcripts for {meeting} in {output_dir}")

    logger.info("All tasks completed successfully!")
