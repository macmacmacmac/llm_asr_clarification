"""
ES2005d top-20 tables for the four teacher metrics (sampled expected score, sampled max, logp 8B,
logp 32B), with the quiz first, a "wrong on baseline" column and a manual "clarification seems
important" judgment. Orderings match the earlier per-metric tables (expected_score_hist.py,
top20_max_metric_table.py, top20_by_method.py).

Baseline correctness per table uses the model behind the metric: Qwen3-8B (sampled base p(correct) < 0.5)
for the three 8B metrics, and the official Qwen3-32B quiz run (vllm_quiz.json) for logp 32B.

JUDGMENTS were made by reading each GT line against the question's gold answer:
  Yes    = the GT line supplies answer-relevant content that the ASR line garbles or omits
  Partly = supplies context for the question, but not the answer itself
  No     = unrelated to the question, or the ASR line already says the same thing

Run from the repo root (meeting defaults to ES2005d; JUDGMENTS only exist for ES2005d):
    python results/sampled_deltas/top20_tables_all_metrics.py [MEETING]
Writes results/sampled_deltas/top20_tables_<MEETING>.md.
"""

import sys
import json

import torch
import numpy as np

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
AMI = "shared/datasets/amicorpus/train"
M = sys.argv[1] if len(sys.argv) > 1 else "ES2005d"

# (line, question) -> judgment
JUDGMENTS = {
    (56, 2): "Yes", (85, 5): "Yes", (106, 6): "Yes", (109, 6): "Yes", (231, 1): "Yes", (232, 1): "Yes",
    (468, 8): "Yes",
    (45, 3): "Partly — sets up that tomatoes weren't discussed", (46, 3): "Partly — sets up that tomatoes weren't discussed",
    (47, 3): "Partly — sets up that tomatoes weren't discussed",
    (48, 3): "No — ASR already says it", (53, 4): "No — ASR already says it", (144, 0): "No — ASR already says it",
    (424, 7): "No — GT line drops the relevant content",
}

d = torch.load(f"{R}/{M}.pt", weights_only=False)
valid = ~d["is_noop"].numpy()
p = d["p_correct"].numpy()
quiz = d["quiz"]
wrong_8b = p[0] < 0.5
with open(f"{AMI}/{M}/quiz/vllm_quiz.json") as f:
    wrong_32b = np.array([not q["score_using_custom_transcript_gt_segments"] for q in json.load(f)])

with open(f"{AMI}/{M}/transcripts/custom_transcript_gt_segments.txt") as f:
    asr = [l.strip().split("]: ", 1)[-1].replace("|", "/") for l in f]
with open(f"{AMI}/{M}/transcripts/parsed_diarized_gt.txt") as f:
    gt = [l.strip().split("]: ", 1)[-1].replace("|", "/") for l in f]


def zscore(x):
    mu, sd = x[valid].mean(0), x[valid].std(0)
    return np.where(valid[:, None], (x - mu) / np.where(sd > 0, sd, np.inf), 0.0)


def max_metric(x):
    """Per-question z across lines, max over questions; ties broken by summed z."""
    z = zscore(x)
    order = np.lexsort((-z.sum(1), -np.where(valid, z.max(1), -np.inf)))[:20]
    return [(j, z[j].max(), int(z[j].argmax())) for j in order]


# Sampled expected score (ties broken randomly, seed 0, as in expected_score_hist.py)
s = d["mean_score"].numpy()[1:]
z_es = np.where(valid, (s - s[valid].mean()) / s[valid].std(), -np.inf)
order = np.lexsort((np.random.default_rng(0).random(len(z_es)), -z_es))[:20]
dp = p[1:] - p[0]
expected = [(j, z_es[j], int(dp[j].argmax())) for j in order]

logp8 = torch.load("shared/datasets/logp_deltas/train_Qwen3-8B-FP8.pt")[M]["logps"].numpy()[1:]
logp32 = torch.load("shared/datasets/logp_deltas/train_32b.pt")[M]["logps"].numpy()[1:]

SAMPLING = ("For each line, the line is replaced with its ground-truth version and Qwen3-8B answers every quiz "
            "question 8 times with sampling; Qwen3-8B grades each answer right or wrong against the gold answer.")
LOGP = ("For each line, the line is replaced with its ground-truth version and we measure how likely {model} finds "
        "the exact gold answer to each question (the log-probability of the gold answer text). Each question's "
        "likelihoods are standardized across all lines, a line's score is its highest standardized value over the "
        "questions, and the question column is the question that value comes from.")
SECTIONS = [
    ("Sampled 8B: expected total score",
     SAMPLING + " A line's score is the expected total quiz score after clarifying it (the number of correct answers "
     "out of 10, averaged over the 8 samples), standardized across all lines. The question column is the question "
     "whose chance of being answered correctly rose the most.",
     expected, wrong_8b, "Qwen3-8B"),
    ("Sampled 8B: max over questions",
     SAMPLING + " For each question, the fraction of the 8 answers graded correct is standardized across all lines. "
     "A line's score is its highest standardized value over the questions, and the question column is the question "
     "that value comes from.",
     max_metric(p[1:]), wrong_8b, "Qwen3-8B"),
    ("Logp 8B: max over questions", LOGP.format(model="Qwen3-8B"), max_metric(logp8), wrong_8b, "Qwen3-8B"),
    ("Logp 32B: max over questions", LOGP.format(model="Qwen3-32B"), max_metric(logp32), wrong_32b, "Qwen3-32B"),
]

md = [f"# {M}: top-20 lines per teacher metric", "",
      "## Quiz questions and gold answers", "",
      "| q | question | gold answer | wrong on baseline (Qwen3-8B) | wrong on baseline (Qwen3-32B) |",
      "|---|---|---|---|---|"]
md += [f"| q{i} | {qq['question']} | {qq['correct_answer']} | {'Yes' if wrong_8b[i] else 'No'} | {'Yes' if wrong_32b[i] else 'No'} |"
       for i, qq in enumerate(quiz)]
md.append("")
for title, description, rows, wrong, model in SECTIONS:
    md += [f"## {title}", "", description, "",
           f"| rank | line | ASR text | GT text | z | question | question wrong on baseline ({model})? |"
           + (" clarification seems important? |" if M == "ES2005d" else ""),
           "|---|---|---|---|---|---|---|" + ("---|" if M == "ES2005d" else "")]
    # The manual JUDGMENTS were only made for ES2005d; other meetings get no judgment column.
    md += [f"| {r} | {j} | {asr[j]} | {gt[j]} | {z:.2f} | q{q} | {'Yes' if wrong[q] else 'No'} |"
           + (f" {JUDGMENTS.get((int(j), q), 'No')} |" if M == "ES2005d" else "")
           for r, (j, z, q) in enumerate(rows, 1)]
    md.append("")

with open(f"results/sampled_deltas/top20_tables_{M}.md", "w") as f:
    f.write("\n".join(md))
