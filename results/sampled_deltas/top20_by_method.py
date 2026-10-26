"""
Ad hoc pilot table: top-20 lines per meeting under each importance method, with the question
each line helps most (argmax z). Methods:
  - sampled 8B: per-question mean/std z of sampled p(correct), max over questions
  - logp 8B / logp 32B: per-question mean/std z of answer logp (as generate_logp_deltas.py)
Ties (common for sampled, since p is in steps of 1/K) are broken by the summed z over questions.
No-op lines (GT == ASR) are excluded for every method.

Run from the repo root:
    python results/sampled_deltas/top20_by_method.py
Writes results/sampled_deltas/top20_by_method.md and prints the compact tables.
"""

import torch
import numpy as np

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
AMI = "shared/datasets/amicorpus/train"
MEETINGS = ["ES2005d", "IN1012", "TS3005d"]
N = 20
LOGP = {"logp 8B": "train_Qwen3-8B-FP8", "logp 32B": "train_32b"}


def zscore(x, valid):
    mu, sd = x[valid].mean(0), x[valid].std(0)
    return np.where(valid[:, None], (x - mu) / np.where(sd > 0, sd, np.inf), 0.0)


def ranked(z, valid, n=N):
    primary = np.where(valid, z.max(1), -np.inf)
    secondary = z.sum(1)
    order = np.lexsort((-secondary, -primary))[:n]
    return [(int(j), int(z[j].argmax()), float(z[j].max())) for j in order]


def text(line, width=70):
    t = line.split("]: ", 1)[-1].replace("|", "/")
    return t if len(t) <= width else t[:width - 3] + "..."


logp_data = {k: torch.load(f"shared/datasets/logp_deltas/{v}.pt") for k, v in LOGP.items()}
md = ["# Top-20 lines per method (pilot, train)", "",
      "Cell = `line (q#) z`. **Bold** = line is in the top 20 of at least one other method.",
      "Sampled: Qwen3-8B, K=8, T=0.7; z = per-question mean/std of p(correct) across lines. "
      "Logp: per-question mean/std of the gold-answer logp across lines. No-op lines excluded.", ""]

for m in MEETINGS:
    d = torch.load(f"{R}/{m}.pt", weights_only=False)
    valid = ~d["is_noop"].numpy()
    tops = {"sampled 8B": ranked(zscore(d["p_correct"].numpy()[1:], valid), valid)}
    for k, ld in logp_data.items():
        tops[k] = ranked(zscore(ld[m]["logps"].numpy()[1:], valid), valid)

    sets = {k: {j for j, _, _ in v} for k, v in tops.items()}
    def cell(k, j, q, zv):
        shared = any(j in s for kk, s in sets.items() if kk != k)
        c = f"{j} (q{q}) {zv:.1f}"
        return f"**{c}**" if shared else c

    with open(f"{AMI}/{m}/transcripts/custom_transcript_gt_segments.txt") as f:
        asr = [l.strip() for l in f]
    with open(f"{AMI}/{m}/transcripts/parsed_diarized_gt.txt") as f:
        gt = [l.strip() for l in f]

    names = list(tops)
    md += [f"## {m}", "", "| rank | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for r in range(N):
        md.append(f"| {r + 1} | " + " | ".join(cell(k, *tops[k][r]) for k in names) + " |")
    md += ["", "Shared lines: " + ", ".join(f"{a} ∩ {b} = {len(sets[a] & sets[b])}" for i, a in enumerate(names) for b in names[i + 1:]), ""]

    union = sorted(set().union(*sets.values()))
    md += ["<details><summary>Line text (union of all three lists)</summary>", "",
           "| line | in | ASR | GT |", "|---|---|---|---|"]
    for j in union:
        md.append(f"| {j} | {', '.join(k for k in names if j in sets[k])} | {text(asr[j])} | {text(gt[j])} |")
    md += ["", "</details>", "", "Questions:", ""]
    md += [f"- q{i}: {q['question']} — *{q['correct_answer']}*" for i, q in enumerate(d["quiz"])]
    md.append("")

out = "results/sampled_deltas/top20_by_method.md"
with open(out, "w") as f:
    f.write("\n".join(md))
print("\n".join(md))
