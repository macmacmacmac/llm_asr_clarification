"""
Ad hoc pilot analysis: robust (median/MAD) vs regular (mean/std) per-question standardization
of sampled p(correct) across lines. Reports top-20 question dominance, split-half stability
(first K/2 vs last K/2 samples), and agreement with the logp-delta z-scores.

Run from the repo root:
    python results/sampled_deltas/compare_standardization.py
"""

import torch
import numpy as np
import pandas as pd

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
MEETINGS = ["ES2005d", "IN1012", "TS3005d"]
LOGP = {n: torch.load(f"shared/datasets/logp_deltas/{n}.pt") for n in ["train_Qwen3-8B-FP8", "train_32b"]}


def robust(p, valid, K):
    med = np.median(p[valid], 0)
    mad = np.median(np.abs(p[valid] - med), 0) * 1.4826
    return np.where(valid[:, None], (p - med) / np.maximum(mad, 1 / K), 0.0)


def regular(p, valid, K):
    mu = p[valid].mean(0)
    sd = p[valid].std(0)
    return np.where(valid[:, None], (p - mu) / np.where(sd > 0, sd, np.inf), 0.0)


def top(score, valid, n=20):
    s = np.where(valid, score, -np.inf)
    # random tie-break so saturated ties don't inherit line order
    return np.lexsort((np.random.default_rng(0).random(len(s)), -s))[:n]


def sp(a, b):
    return round(float(pd.Series(a).corr(pd.Series(b), method="spearman")), 3)


rows = []
for m in MEETINGS:
    d = torch.load(f"{R}/{m}.pt", weights_only=False)
    c = d["correct"].numpy()[1:].astype(float)
    valid = ~d["is_noop"].numpy()
    K = c.shape[2]
    for name, f in [("robust", robust), ("regular", regular)]:
        z = f(c.mean(-1), valid, K)
        score = z.max(1)
        t = top(score, valid)
        dom = np.bincount(z[t].argmax(1), minlength=z.shape[1])
        sa = f(c[..., :K // 2].mean(-1), valid, K // 2).max(1)
        sb = f(c[..., K // 2:].mean(-1), valid, K // 2).max(1)
        row = {
            "meeting": m,
            "method": name,
            "max q share": int(dom.max()),
            "#q in top20": int((dom > 0).sum()),
            "half spearman": sp(sa[valid], sb[valid]),
            "half top20": len(set(top(sa, valid)) & set(top(sb, valid))),
        }
        for ln, ld in LOGP.items():
            L = ld[m]["logps"].numpy()[1:]
            ls = ((L - L.mean(0)) / L.std(0)).max(1)
            tag = "8b" if "8B" in ln else "32b"
            row[f"sp vs logp{tag}"] = sp(score[valid], ls[valid])
            row[f"top20 vs logp{tag}"] = len(set(t) & set(top(ls, valid)))
        row["top20 per-q counts"] = dom.tolist()
        rows.append(row)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))
