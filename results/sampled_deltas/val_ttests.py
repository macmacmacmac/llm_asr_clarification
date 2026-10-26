"""
One-sided paired t-tests, H1: mean(LOGP32B - COSINE) > 0, on the fresh validation rerun (tag _r2),
at each clarification budget K. Two pairings:
  - per question (n = 180): difference of the 0/1 grades
  - per meeting (n = 18): difference of the meeting's quiz accuracy (questions within a meeting are not
    independent, so this is the more conservative test)

Run from the repo root (.vllm_venv has scipy):
    python results/sampled_deltas/val_ttests.py
"""

import os
import json

import numpy as np
from scipy import stats

AMI = "shared/datasets/amicorpus/validation"
KS = [10, 20, 30, 40, 50]


def field(key, k):
    return f"score_using_custom_transcript_gt_segments_all_{key}_r2_{k}_clarify4"


quizzes = {}
for m in sorted(os.listdir(AMI)):
    with open(f"{AMI}/{m}/quiz/vllm_quiz.json") as f:
        quizzes[m] = json.load(f)

print(f"{'K':>3} | {'per question (n=180)':^36} | {'per meeting (n=18)':^36}")
print(f"{'':>3} | {'mean diff':>10} {'t':>8} {'p (1-sided)':>14} | {'mean diff':>10} {'t':>8} {'p (1-sided)':>14}")
for k in KS:
    q_diff = np.array([q[field("logp32b", k)] - q[field("gt", k)] for quiz in quizzes.values() for q in quiz], dtype=float)
    m_diff = np.array([np.mean([q[field("logp32b", k)] - q[field("gt", k)] for q in quiz]) for quiz in quizzes.values()])
    tq = stats.ttest_1samp(q_diff, 0.0, alternative="greater")
    tm = stats.ttest_1samp(m_diff, 0.0, alternative="greater")
    print(f"{k:>3} | {q_diff.mean():>10.3f} {tq.statistic:>8.2f} {tq.pvalue:>14.3f} | {m_diff.mean():>10.3f} {tm.statistic:>8.2f} {tm.pvalue:>14.3f}")

# Pooled over K (post hoc): one number per meeting = its accuracy difference averaged over the 5 budgets
# (area under the accuracy-vs-K curve). The K rows share questions and nested line sets, so they are not
# stacked as independent observations; the meeting is the unit (n = 18).
pooled = np.array([np.mean([np.mean([q[field("logp32b", k)] - q[field("gt", k)] for q in quiz]) for k in KS])
                   for quiz in quizzes.values()])
tp = stats.ttest_1samp(pooled, 0.0, alternative="greater")
wins, losses = int((pooled > 0).sum()), int((pooled < 0).sum())
print(f"\nPooled over K, per meeting (n=18): mean diff {pooled.mean():.3f}, t = {tp.statistic:.2f}, one-sided p = {tp.pvalue:.3f}; "
      f"LOGP32B ahead in {wins} meetings, behind in {losses}, tied in {len(pooled) - wins - losses}")
