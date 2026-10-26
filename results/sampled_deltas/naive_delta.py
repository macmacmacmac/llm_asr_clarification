"""
Ad hoc pilot analysis: naive marginal gain of sampled p(correct) over the base transcript,
    delta[j, q] = p[j, q] - p[base, q],  line score = max_q delta,  attributed question = argmax_q,
compared with the mean/std z-scored version and logp 32B on: top-20 question histogram, ties at
the top, split-half stability (first K/2 vs last K/2 samples), overlap with logp 32B, and the
optimistic coverage test against the official baseline (fixable questions only).

Run from the repo root:
    python results/sampled_deltas/naive_delta.py
"""

import json

import torch
import numpy as np
import pandas as pd

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
AMI = "shared/datasets/amicorpus/train"
MEETINGS = ["ES2005d", "IN1012", "TS3005d"]
BUDGETS = [3, 5, 10, 20]


def zscore(x, valid):
    mu, sd = x[valid].mean(0), x[valid].std(0)
    return np.where(valid[:, None], (x - mu) / np.where(sd > 0, sd, np.inf), 0.0)


def naive(p_lines, p_base, valid):
    return np.where(valid[:, None], p_lines - p_base[None, :], -np.inf)


def ranked(s, valid, n):
    """Top-n line indices by max over questions, ties broken by summed score."""
    finite = np.where(np.isfinite(s), s, 0.0)
    order = np.lexsort((-finite.sum(1), -np.where(valid, s.max(1), -np.inf)))
    return order[:n]


def sp(a, b):
    return round(float(pd.Series(a).corr(pd.Series(b), method="spearman")), 3)


logp32 = torch.load("shared/datasets/logp_deltas/train_32b.pt")
rows, top_lines = [], {}

for m in MEETINGS:
    d = torch.load(f"{R}/{m}.pt", weights_only=False)
    c = d["correct"].numpy().astype(float)
    p = c.mean(-1)
    valid = ~d["is_noop"].numpy()
    K = c.shape[2]

    with open(f"{AMI}/{m}/quiz/vllm_quiz.json") as f:
        quiz = json.load(f)
    wrong = ~np.array([q["score_using_custom_transcript_gt_segments"] for q in quiz]).astype(bool)
    gt_ok = np.array([q["score_using_parsed_diarized_gt"] for q in quiz]).astype(bool)
    fixable = set(np.nonzero(wrong & gt_ok)[0].tolist())

    L = logp32[m]["logps"].numpy()[1:]
    lz = zscore(L, valid)
    top32 = set(ranked(lz, valid, 20).tolist())

    pa, pb = c[..., :K // 2].mean(-1), c[..., K // 2:].mean(-1)
    methods = {
        "naive delta": (naive(p[1:], p[0], valid), naive(pa[1:], pa[0], valid), naive(pb[1:], pb[0], valid)),
        "z (mean/std)": (zscore(p[1:], valid), zscore(pa[1:], valid), zscore(pb[1:], valid)),
    }
    for name, (s, sa, sb) in methods.items():
        top = ranked(s, valid, 20)
        top_score = s[top[0]].max()
        hist = np.bincount(s[top].argmax(1), minlength=s.shape[1])
        row = {
            "meeting": m, "method": name,
            "top20 per-q counts": hist.tolist(),
            "lines tied at top score": int((np.where(valid, s.max(1), -np.inf) == top_score).sum()),
            "lines with max delta>0" if name == "naive delta" else "lines with max z>0": int((np.where(valid, s.max(1), -np.inf) > 0).sum()),
            "half spearman": sp(np.where(valid, sa.max(1), 0)[valid], np.where(valid, sb.max(1), 0)[valid]),
            "half top20": len(set(ranked(sa, valid, 20)) & set(ranked(sb, valid, 20))),
            "top20 ∩ logp32B": len(set(top.tolist()) & top32),
        }
        for k in BUDGETS:
            row[f"fix@{k}"] = len(set(s[ranked(s, valid, k)].argmax(1).tolist()) & fixable)
        rows.append(row)
        if name == "naive delta":
            top_lines[m] = [(int(j), int(s[j].argmax()), round(float(p[0, s[j].argmax()]), 2), round(float(p[j + 1, s[j].argmax()]), 2)) for j in top[:10]]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))
print("\nNaive delta top-10 lines: (line, q, p base, p line)")
for m, tl in top_lines.items():
    print(f"  {m}: {tl}")
