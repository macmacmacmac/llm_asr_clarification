"""
Ad hoc pilot analysis, two parts:

1. Histogram of which question each top-20 line helps (argmax-z question), per method and meeting.
   Saved as results/sampled_deltas/top20_question_hist.png.

2. Cheap optimistic coverage test: "clarify" each method's top-k lines and assume any clarified
   line attributed to question q flips q from incorrect to correct. Gain = number of
   baseline-wrong questions covered. Baselines:
     - official: score_using_custom_transcript_gt_segments in vllm_quiz.json (greedy quiz run)
     - sampled 8B: base p(correct) < 0.5 from the K=8 sampled run
   "Fixable" questions are those the GT transcript gets right (score_using_parsed_diarized_gt);
   a question the GT transcript also fails can't be fixed by clarification.

Run from the repo root:
    python results/sampled_deltas/coverage_test.py
"""

import json

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
AMI = "shared/datasets/amicorpus/train"
MEETINGS = ["ES2005d", "IN1012", "TS3005d"]
METHODS = ["sampled 8B", "logp 8B", "logp 32B"]
LOGP = {"logp 8B": "train_Qwen3-8B-FP8", "logp 32B": "train_32b"}
COLORS = {"sampled 8B": "#2a78d6", "logp 8B": "#eb6834", "logp 32B": "#1baf7a"}  # categorical slots 1-3
BUDGETS = [3, 5, 10, 20]


def zscore(x, valid):
    mu, sd = x[valid].mean(0), x[valid].std(0)
    return np.where(valid[:, None], (x - mu) / np.where(sd > 0, sd, np.inf), 0.0)


def ranked_q(z, valid, n):
    """Argmax question of the top-n lines (ties broken by summed z, as in top20_by_method.py)."""
    order = np.lexsort((-z.sum(1), -np.where(valid, z.max(1), -np.inf)))[:n]
    return z[order].argmax(1)


logp_data = {k: torch.load(f"shared/datasets/logp_deltas/{v}.pt") for k, v in LOGP.items()}
hist, cov_rows, base_rows, official_wrong, sampled_wrong = {}, [], [], {}, {}

for m in MEETINGS:
    d = torch.load(f"{R}/{m}.pt", weights_only=False)
    valid = ~d["is_noop"].numpy()
    num_q = d["p_correct"].shape[1]
    zs = {"sampled 8B": zscore(d["p_correct"].numpy()[1:], valid)}
    for k, ld in logp_data.items():
        zs[k] = zscore(ld[m]["logps"].numpy()[1:], valid)

    with open(f"{AMI}/{m}/quiz/vllm_quiz.json") as f:
        quiz = json.load(f)
    official = np.array([q["score_using_custom_transcript_gt_segments"] for q in quiz])
    gt_ok = np.array([q["score_using_parsed_diarized_gt"] for q in quiz]).astype(bool)
    baselines = {"official": ~official.astype(bool), "sampled 8B": d["p_correct"].numpy()[0] < 0.5}
    official_wrong[m] = baselines["official"]
    sampled_wrong[m] = baselines["sampled 8B"]
    print(f"{m} 8B sampled base p: {np.round(d['p_correct'].numpy()[0], 3).tolist()}")

    for bname, wrong in baselines.items():
        base_rows.append({
            "meeting": m, "baseline": bname,
            "score": int((~wrong).sum()),
            "wrong qs": [int(q) for q in np.nonzero(wrong)[0]],
            "fixable wrong qs (GT gets right)": [int(q) for q in np.nonzero(wrong & gt_ok)[0]],
        })

    for meth in METHODS:
        hist[(m, meth)] = np.bincount(ranked_q(zs[meth], valid, 20), minlength=num_q)
        for k in BUDGETS:
            covered = set(ranked_q(zs[meth], valid, k).tolist())
            row = {"meeting": m, "method": meth, "top-k": k, "qs covered": sorted(covered)}
            for bname, wrong in baselines.items():
                wrong_qs = set(np.nonzero(wrong)[0].tolist())
                row[f"gain ({bname})"] = len(covered & wrong_qs)
                row[f"of {bname} wrong"] = len(wrong_qs)
            cov_rows.append(row)

# 1. Histogram figure: one panel per meeting, grouped bars per question
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
width = 0.27
for ax, m in zip(axes, MEETINGS):
    x = np.arange(10)
    for i, meth in enumerate(METHODS):
        ax.bar(x + (i - 1) * width, hist[(m, meth)], width=width - 0.03, color=COLORS[meth], label=meth)
    ax.set_title(m)
    ax.set_xticks(x, [f"q{i}*" if sampled_wrong[m][i] else f"q{i}" for i in x])
    ax.set_xlabel("question")
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("lines in top 20")
axes[0].legend(frameon=False)
fig.suptitle("Which question each method's top-20 lines help (argmax-z question)")
fig.text(0.5, 0.005, "* = question Qwen3-8B gets wrong on the original ASR transcript "
         "(approximated as sampled base p(correct) < 0.5, K=8, T=0.7)", ha="center", fontsize=9, color="#5f5e5a")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("results/sampled_deltas/top20_question_hist.png", dpi=150)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)
print("Top-20 question histogram (counts q0..q9):")
for m in MEETINGS:
    for meth in METHODS:
        print(f"  {m:8s} {meth:11s} {hist[(m, meth)].tolist()}")
print()
print(pd.DataFrame(base_rows).to_string(index=False))
print()
print(pd.DataFrame(cov_rows).to_string(index=False))
