"""
Test-retest reliability of candidate regression targets built from the 32B logp deltas, using ES2005d
computed under three execution paths: repro run (8192-token steps), full train run (16384), and the old
pilot file. For each target definition, the correlation between two paths is the noise ceiling: no
student can correlate with one run's targets better (in expectation) than another run does.

Targets (per line j, per question q unless noted; computed over lines whose GT text differs from ASR):
  delta_vs_base    logp[j, q] - logp[base, q]
  z_per_q          (logp[j, q] - mean_j) / std_j            (as in generate_logp_deltas / the detectors)
  line_max_z       max_q z_per_q                            (one value per line; what pipelinev4 ranks by)
  line_sum_delta   sum_q (logp[j, q] - logp[base, q])       (one value per line)

Run from the repo root:
    python results/sampled_deltas/logp_target_reliability.py
"""

from itertools import combinations

import torch
import numpy as np
import pandas as pd

M = "ES2005d"
AMI = "shared/datasets/amicorpus/train"
SOURCES = {
    "repro (8192)": "shared/datasets/logp_deltas/full/repro/ES2005d_run1.pt",
    "full run (16384)": "shared/datasets/logp_deltas/full/train_Qwen3-32B-FP8.pt",
    "old pilot file": "shared/datasets/logp_deltas/train_32b.pt",
}

with open(f"{AMI}/{M}/transcripts/custom_transcript_gt_segments.txt") as f:
    asr = [l.strip() for l in f]
with open(f"{AMI}/{M}/transcripts/parsed_diarized_gt.txt") as f:
    gt = [l.strip() for l in f]
valid = np.array([a != g for a, g in zip(asr, gt)])

targets = {}
for name, path in SOURCES.items():
    L = torch.load(path)[M]["logps"].numpy()
    base, lines = L[0], L[1:][valid]
    z = (lines - lines.mean(0)) / lines.std(0)
    targets[name] = {
        "delta_vs_base": (lines - base).ravel(),
        "z_per_q": z.ravel(),
        "line_max_z": z.max(1),
        "line_sum_delta": (lines - base).sum(1),
    }

rows = []
for target in ["delta_vs_base", "z_per_q", "line_max_z", "line_sum_delta"]:
    for a, b in combinations(SOURCES, 2):
        x, y = targets[a][target], targets[b][target]
        rows.append({"target": target, "pair": f"{a} vs {b}",
                     "pearson": round(float(np.corrcoef(x, y)[0, 1]), 3),
                     "spearman": round(float(pd.Series(x).corr(pd.Series(y), method="spearman")), 3)})
pd.set_option("display.width", 200)
print(f"{M}: {valid.sum()} lines with GT != ASR\n")
print(pd.DataFrame(rows).to_string(index=False))
