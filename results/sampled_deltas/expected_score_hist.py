"""
Ad hoc pilot figure + table for the expected-total-score metric (supervisor's version):
    score[j] = mean over K samples of the summed 0/1 grades with only line j clarified (= mean_score[j+1]),
    z[j] = (score[j] - mean over lines) / std over lines   (no-op lines excluded).
The metric has no question of its own; a line is attributed to the question whose p(correct) rose most
vs the base transcript (argmax_q p[j, q] - p[base, q]), i.e. the biggest contributor to its gain.

Outputs:
  results/sampled_deltas/top20_question_hist_expected_score.png  (same layout as top20_question_hist.png,
      sampled bars replaced by this metric; * = 8B gets it wrong on the base transcript, p < 0.5)
  results/sampled_deltas/top20_expected_score_ES2005d.md          (top-20 table for ES2005d)

Run from the repo root:
    python results/sampled_deltas/expected_score_hist.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
AMI = "shared/datasets/amicorpus/train"
MEETINGS = ["ES2005d", "IN1012", "TS3005d"]
LOGP = {"logp 8B": "train_Qwen3-8B-FP8", "logp 32B": "train_32b"}
METHODS = ["expected score (8B)", "logp 8B", "logp 32B"]
COLORS = {"expected score (8B)": "#2a78d6", "logp 8B": "#eb6834", "logp 32B": "#1baf7a"}  # categorical slots 1-3
TABLE_MEETING = "ES2005d"


def zscore(x, valid):
    mu, sd = x[valid].mean(0), x[valid].std(0)
    return np.where(valid[:, None], (x - mu) / np.where(sd > 0, sd, np.inf), 0.0)


def top20(score, valid, seed=0):
    s = np.where(valid, score, -np.inf)
    return np.lexsort((np.random.default_rng(seed).random(len(s)), -s))[:20]


def text(line):
    return line.split("]: ", 1)[-1].replace("|", "/")


logp_data = {k: torch.load(f"shared/datasets/logp_deltas/{v}.pt") for k, v in LOGP.items()}
hist, wrong8b = {}, {}

for m in MEETINGS:
    d = torch.load(f"{R}/{m}.pt", weights_only=False)
    valid = ~d["is_noop"].numpy()
    p = d["p_correct"].numpy()
    num_q = p.shape[1]
    wrong8b[m] = p[0] < 0.5

    s = d["mean_score"].numpy()[1:]
    z = np.where(valid, (s - s[valid].mean()) / s[valid].std(), -np.inf)
    dp = p[1:] - p[0]                                  # [N, Q] change vs base
    t = top20(z, valid)
    hist[(m, METHODS[0])] = np.bincount(dp[t].argmax(1), minlength=num_q)
    for name, key in LOGP.items():
        L = logp_data[name][m]["logps"].numpy()[1:]
        lz = zscore(L, valid)
        lt = top20(lz.max(1), valid)
        hist[(m, name)] = np.bincount(lz[lt].argmax(1), minlength=num_q)

    if m == TABLE_MEETING:
        with open(f"{AMI}/{m}/transcripts/custom_transcript_gt_segments.txt") as f:
            asr = [l.strip() for l in f]
        with open(f"{AMI}/{m}/transcripts/parsed_diarized_gt.txt") as f:
            gt = [l.strip() for l in f]
        md = [
            f"# {m}: top-20 lines by expected total quiz score (Qwen3-8B, K=8)",
            "",
            f"Base transcript expected score: {d['mean_score'][0]:.2f}/10. Score = expected total score with only "
            "that line clarified; z = standardized across lines (no-op lines excluded). "
            "Main question = largest rise in p(correct) vs base; all questions moved by >= 0.25 listed with p base → p line.",
            "",
            "| rank | line | z | score | main q | all moved questions | ASR | GT |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r, j in enumerate(t, 1):
            q = int(dp[j].argmax())
            moved = ", ".join(f"q{k}{'*' if wrong8b[m][k] else ''} {p[0, k]:.2f}→{p[j + 1, k]:.2f}"
                              for k in np.nonzero(np.abs(dp[j]) >= 0.25)[0])
            md.append(f"| {r} | {j} | {z[j]:.2f} | {s[j]:.2f} | q{q}{'*' if wrong8b[m][q] else ''} | {moved} | "
                      f"{text(asr[j])} | {text(gt[j])} |")
        md += ["", "\\* = Qwen3-8B gets this question wrong on the base transcript (sampled p < 0.5).", "", "Questions:", ""]
        md += [f"- q{i}{'*' if wrong8b[m][i] else ''}: {qq['question']} — *{qq['correct_answer']}*" for i, qq in enumerate(d["quiz"])]
        with open(f"results/sampled_deltas/top20_expected_score_{m}.md", "w") as f:
            f.write("\n".join(md))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
width = 0.27
for ax, m in zip(axes, MEETINGS):
    x = np.arange(10)
    for i, meth in enumerate(METHODS):
        ax.bar(x + (i - 1) * width, hist[(m, meth)], width=width - 0.03, color=COLORS[meth], label=meth)
    ax.set_title(m)
    ax.set_xticks(x, [f"q{i}*" if wrong8b[m][i] else f"q{i}" for i in x])
    ax.set_xlabel("question")
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("lines in top 20")
axes[0].legend(frameon=False)
fig.suptitle("Which question each method's top-20 lines help (expected score: largest p rise vs base; logp: argmax-z question)")
fig.text(0.5, 0.005, "* = question Qwen3-8B gets wrong on the original ASR transcript "
         "(approximated as sampled base p(correct) < 0.5, K=8, T=0.7)", ha="center", fontsize=9, color="#5f5e5a")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("results/sampled_deltas/top20_question_hist_expected_score.png", dpi=150)

for m in MEETINGS:
    for meth in METHODS:
        print(f"{m:8s} {meth:20s} {hist[(m, meth)].tolist()}")
