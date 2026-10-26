"""
Ad hoc pilot check: where do the clean single-line sampled wins rank under the logp z-scores
(overall by max-over-questions, and within their own question)?

Run from the repo root:
    python results/sampled_deltas/clean_wins_in_logp.py
"""

import torch
import numpy as np

WINS = [("IN1012", 917, 9), ("IN1012", 36, 0), ("TS3005d", 36, 1), ("ES2005d", 106, 6)]  # (meeting, line, question)

for name in ["train_Qwen3-8B-FP8", "train_32b"]:
    d = torch.load(f"shared/datasets/logp_deltas/{name}.pt")
    print(f"== {name}")
    for m, j, q in WINS:
        L = d[m]["logps"].numpy()[1:]
        z = (L - L.mean(0)) / L.std(0)
        overall = int((z.max(1) > z[j].max()).sum()) + 1
        within_q = int((z[:, q] > z[j, q]).sum()) + 1
        print(f"  {m} line {j} (q{q}): z={z[j, q]:.1f}, rank within q{q} = {within_q}, overall rank = {overall} / {len(z)}")
