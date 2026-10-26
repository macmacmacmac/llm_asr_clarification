"""
Compare repeated 32B logp-delta computations of ES2005d: the two runs from logp_repro.sh, the full train
run (logp_deltas_full.sh), and the earlier pilot file (train_32b.pt). For each pair: max / median absolute
difference of the raw logps, Spearman of the line scores (max over questions of the per-question z), and
top-20 overlap.

Run from the repo root after logp_repro.sh:
    python results/sampled_deltas/logp_repro_compare.py
"""

from itertools import combinations

import torch
import numpy as np
import pandas as pd

M = "ES2005d"
SOURCES = {
    "repro run1": "shared/datasets/logp_deltas/full/repro/ES2005d_run1.pt",
    "repro run2": "shared/datasets/logp_deltas/full/repro/ES2005d_run2.pt",
    "full train run": "shared/datasets/logp_deltas/full/train_Qwen3-32B-FP8.pt",
    "old pilot file": "shared/datasets/logp_deltas/train_32b.pt",
}

logps = {name: torch.load(path)[M]["logps"].numpy() for name, path in SOURCES.items()}


def line_score(L):
    v = L[1:]
    return ((v - v.mean(0)) / v.std(0)).max(1)


rows = []
for a, b in combinations(SOURCES, 2):
    d = np.abs(logps[a] - logps[b])
    sa, sb = line_score(logps[a]), line_score(logps[b])
    rows.append({
        "pair": f"{a} vs {b}",
        "identical": bool((d == 0).all()),
        "max |diff|": round(float(d.max()), 4),
        "median |diff|": round(float(np.median(d)), 4),
        "line-score spearman": round(float(pd.Series(sa).corr(pd.Series(sb), method="spearman")), 3),
        "top-20 overlap": len(set(np.argsort(-sa)[:20]) & set(np.argsort(-sb)[:20])),
    })
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
