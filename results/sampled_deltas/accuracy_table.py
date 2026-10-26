"""
Average quiz accuracy (correct answers / questions, pooled over meetings) for each teacher at each
clarification budget K, in the Slack-friendly layout used for earlier pipelinev4 results
("<mistranscription detector> + <importance detector>"). Random is the mean over its 3 seeds.

Run from the repo root:
    python results/sampled_deltas/accuracy_table.py
"""

import json

import numpy as np

AMI = "shared/datasets/amicorpus/train"
KS = [10, 20, 30, 40, 50]
METHODS = [("ALL + GT", ["gt"]), ("ALL + SAMPLED", ["sampled"]), ("ALL + LOGP8B", ["logp8b"]),
           ("ALL + LOGP32B", ["logp32b"]), ("ALL + RANDOM", ["random", "random_s1", "random_s2"])]
FIELD = "score_using_custom_transcript_gt_segments{}"


def load(meetings):
    quizzes = {}
    for m in meetings:
        with open(f"{AMI}/{m}/quiz/vllm_quiz.json") as f:
            quizzes[m] = json.load(f)
    return quizzes


def accuracy(quizzes, suffix):
    scores = [q[FIELD.format(suffix)] for quiz in quizzes.values() for q in quiz if FIELD.format(suffix) in q]
    n = sum(len(quiz) for quiz in quizzes.values())
    return sum(scores) / n if len(scores) == n else None


def table(meetings):
    quizzes = load(meetings)
    lines = [f"{'':14s}" + "".join(f"{k:>7d}" for k in KS)]
    for label, keys in METHODS:
        accs = []
        for k in KS:
            vals = [accuracy(quizzes, f"_all_{key}_{k}_clarify4") for key in keys]
            accs.append(None if any(v is None for v in vals) else float(np.mean(vals)))
        if all(a is None for a in accs):
            continue
        lines.append(f"{label:14s}" + "".join(f"{a:7.3f}" if a is not None else f"{'n/a':>7s}" for a in accs))
    asr = accuracy(quizzes, "")
    gt = sum(q["score_using_parsed_diarized_gt"] for quiz in quizzes.values() for q in quiz) / sum(len(q) for q in quizzes.values())
    lines.append(f"{'ASR (K=0)':14s}{asr:7.3f}")
    lines.append(f"{'GT transcript':14s}{gt:7.3f}")
    return "\n".join(lines)


print("3 meetings (ES2005d, IN1012, TS3005d; 30 questions):")
print(table(["ES2005d", "IN1012", "TS3005d"]))
print()
print("4 meetings (+ IS1007c; 40 questions; no SAMPLED labels for IS1007c):")
print(table(["ES2005d", "IN1012", "IS1007c", "TS3005d"]))
