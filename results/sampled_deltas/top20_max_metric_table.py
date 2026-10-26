"""
Ad hoc table: ES2005d top-20 lines under the old sampled "max" metric,
    z[j, q] = (p[j, q] - mean_j p[:, q]) / std_j p[:, q]   (per question, across lines, no-op lines excluded)
    line score = max_q z[j, q],  question = argmax_q
with the same columns as top20_expected_score_ES2005d.md. Ties broken by summed z (as top20_by_method.py).

Run from the repo root:
    python results/sampled_deltas/top20_max_metric_table.py
Writes results/sampled_deltas/top20_max_metric_ES2005d.md (and prints it).
"""

import torch
import numpy as np

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
AMI = "shared/datasets/amicorpus/train"
M = "ES2005d"

d = torch.load(f"{R}/{M}.pt", weights_only=False)
valid = ~d["is_noop"].numpy()
p = d["p_correct"].numpy()
wrong = p[0] < 0.5

lines = p[1:]
mu, sd = lines[valid].mean(0), lines[valid].std(0)
z = np.where(valid[:, None], (lines - mu) / np.where(sd > 0, sd, np.inf), 0.0)
score = np.where(valid, z.max(1), -np.inf)
top = np.lexsort((-z.sum(1), -score))[:20]
dp = lines - p[0]

with open(f"{AMI}/{M}/transcripts/custom_transcript_gt_segments.txt") as f:
    asr = [l.strip().split("]: ", 1)[-1].replace("|", "/") for l in f]
with open(f"{AMI}/{M}/transcripts/parsed_diarized_gt.txt") as f:
    gt = [l.strip().split("]: ", 1)[-1].replace("|", "/") for l in f]


def star(q):
    return f"q{q}{'*' if wrong[q] else ''}"


md = [
    f"# {M}: top-20 lines by the sampled max metric (Qwen3-8B, K=8)",
    "",
    "z = per-question mean/std z of p(correct) across lines; line score = max over questions; question = argmax. "
    "All questions moved by >= 0.25 listed with p base → p line.",
    "",
    "| rank | line | z | question | p base → p line | all moved questions | ASR | GT |",
    "|---|---|---|---|---|---|---|---|",
]
for r, j in enumerate(top, 1):
    q = int(z[j].argmax())
    moved = ", ".join(f"{star(k)} {p[0, k]:.2f}→{p[j + 1, k]:.2f}" for k in np.nonzero(np.abs(dp[j]) >= 0.25)[0])
    md.append(f"| {r} | {j} | {z[j, q]:.2f} | {star(q)} | {p[0, q]:.2f}→{p[j + 1, q]:.2f} | {moved} | {asr[j]} | {gt[j]} |")
md += ["", "\\* = Qwen3-8B gets this question wrong on the base transcript (sampled p < 0.5)."]

with open(f"results/sampled_deltas/top20_max_metric_{M}.md", "w") as f:
    f.write("\n".join(md))
print("\n".join(md))
