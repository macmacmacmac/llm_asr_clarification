"""
Ad hoc pilot check of the supervisor's metric: per line j, expected total quiz score after clarifying
line j (sum over questions of the 0/1 grades, averaged over the K samples = mean_score[j+1]),
standardized across lines. Reports ties, sign of gains vs the base, split-half stability, overlap
with logp 32B, and the top-10 lines with the questions that moved.

Run from the repo root:
    python results/sampled_deltas/expected_score_metric.py
"""

import torch
import numpy as np
import pandas as pd

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
AMI = "shared/datasets/amicorpus/train"
MEETINGS = ["ES2005d", "IN1012", "TS3005d"]


def top(score, valid, n=20, seed=0):
    s = np.where(valid, score, -np.inf)
    return np.lexsort((np.random.default_rng(seed).random(len(s)), -s))[:n]


def sp(a, b):
    return round(float(pd.Series(a).corr(pd.Series(b), method="spearman")), 3)


logp32 = torch.load("shared/datasets/logp_deltas/train_32b.pt")
rows = []
for m in MEETINGS:
    d = torch.load(f"{R}/{m}.pt", weights_only=False)
    c = d["correct"].numpy().astype(float)          # [V, Q, K]
    valid = ~d["is_noop"].numpy()
    K = c.shape[2]
    total = c.sum(1).mean(-1)                       # [V] expected total score (== mean_score)
    s = total[1:]
    z = (s - s[valid].mean()) / s[valid].std()
    gain = s - total[0]

    sa = c[1:, :, :K // 2].sum(1).mean(-1)
    sb = c[1:, :, K // 2:].sum(1).mean(-1)
    L = logp32[m]["logps"].numpy()[1:]
    lz = ((L - L[valid].mean(0)) / L[valid].std(0)).max(1)

    t = top(z, valid)
    rows.append({
        "meeting": m,
        "base score": round(float(total[0]), 2),
        "mean line score": round(float(s[valid].mean()), 2),
        "std across lines": round(float(s[valid].std()), 3),
        "distinct values": int(len(np.unique(s[valid]))),
        "lines tied at max": int((s[valid] == s[valid].max()).sum()),
        "frac lines gain>0": round(float((gain[valid] > 0).mean()), 2),
        "frac lines gain<0": round(float((gain[valid] < 0).mean()), 2),
        "max z": round(float(z[valid].max()), 2),
        "half spearman": sp(sa[valid], sb[valid]),
        "half top20": len(set(top(sa, valid)) & set(top(sb, valid))),
        "top20 ∩ logp32B": len(set(t) & set(top(lz, valid))),
    })

    with open(f"{AMI}/{m}/transcripts/custom_transcript_gt_segments.txt") as f:
        asr = [l.strip().split("]: ", 1)[-1] for l in f]
    with open(f"{AMI}/{m}/transcripts/parsed_diarized_gt.txt") as f:
        gt = [l.strip().split("]: ", 1)[-1] for l in f]
    p = c.mean(-1)
    print(f"== {m} top 10 by expected total score (line, z, score, per-q change vs base)")
    for j in t[:10]:
        dq = p[j + 1] - p[0]
        moved = ", ".join(f"q{q}{dq[q]:+.2f}" for q in np.nonzero(np.abs(dq) >= 0.25)[0])
        print(f"  {j:5d} z={z[j]:5.2f} score={s[j]:.2f} [{moved}] | {asr[j][:55]!r} -> {gt[j][:55]!r}")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print()
print(pd.DataFrame(rows).to_string(index=False))
