"""
Quick-and-dirty teacher (importance) metric comparison. Reads the official quiz results that
quiz.bulk_vllm_offline_quiz wrote into each meeting's vllm_quiz.json for the pipelinev4 transcripts
custom_transcript_gt_segments_all_<imp>_<N>_clarify4, and compares them with the unclarified ASR
transcript (lower bound) and the GT transcript (upper bound).

Sampled deltas only exist for 3 of the 4 meetings, so totals are reported both over all 4 meetings
(sampled column missing) and over the 3 common meetings (all methods comparable).

Run from the repo root after method_comparison.sh finishes:
    python results/sampled_deltas/compare_methods.py
Writes results/sampled_deltas/method_comparison.md (and prints it).
"""

import json

import numpy as np
import pandas as pd

AMI = "shared/datasets/amicorpus/train"
MEETINGS = ["ES2005d", "IN1012", "IS1007c", "TS3005d"]
COMMON = ["ES2005d", "IN1012", "TS3005d"]
METHODS = {"gt": "cosine (previous)", "sampled": "sampled 8B (expected score)", "logp8b": "logp 8B", "logp32b": "logp 32B",
           "random": "random s47", "random_s1": "random s1", "random_s2": "random s2"}
NUM_LINES = [10, 20, 30, 40, 50]
ASR, GT = "score_using_custom_transcript_gt_segments", "score_using_parsed_diarized_gt"

rows = []
for m in MEETINGS:
    with open(f"{AMI}/{m}/quiz/vllm_quiz.json") as f:
        quiz = json.load(f)
    asr = np.array([q[ASR] for q in quiz])
    gt = np.array([q[GT] for q in quiz])
    for key, label in METHODS.items():
        for n in NUM_LINES:
            field = f"score_using_custom_transcript_gt_segments_all_{key}_{n}_clarify4"
            if not all(field in q for q in quiz):
                continue
            s = np.array([q[field] for q in quiz])
            rows.append({
                "meeting": m, "method": label, "N": n, "score": int(s.sum()),
                "asr": int(asr.sum()), "gt": int(gt.sum()),
                "fixed": int(((asr == 0) & (s == 1)).sum()),
                "fixed (GT-fixable)": int(((asr == 0) & (gt == 1) & (s == 1)).sum()),
                "broken": int(((asr == 1) & (s == 0)).sum()),
                "fixed qs": [int(i) for i in np.nonzero((asr == 0) & (s == 1))[0]],
                "broken qs": [int(i) for i in np.nonzero((asr == 1) & (s == 0))[0]],
            })

df = pd.DataFrame(rows)
order = list(METHODS.values())


def totals(meetings):
    sub = df[df.meeting.isin(meetings)]
    t = sub.groupby(["method", "N"])[["score", "asr", "gt", "fixed", "fixed (GT-fixable)", "broken"]].sum()
    count = sub.groupby(["method", "N"]).meeting.nunique().rename("meetings")
    t = t.join(count)
    t = t[t.meetings == len(meetings)]
    t["gain vs ASR"] = t["score"] - t["asr"]
    t = t.reset_index()
    t["method"] = pd.Categorical(t["method"], order, ordered=True)
    return t.sort_values(["N", "method"])


def wide(t, col):
    w = t.pivot(index="method", columns="N", values=col)
    return w.reindex([m for m in order if m in w.index])


md = ["# Teacher-metric comparison (train pilot meetings)", "",
      "Each method ranks all lines (mistranscription detector = ALL, so ranking is importance only); the top N are",
      "replaced with GT by `clarifications.pipelinev4`; the quiz is answered + graded by Qwen3-32B-FP8 greedy",
      "(`quiz.bulk_vllm_offline_quiz`, the same setup as the official ASR/GT scores). 10 questions per meeting.", ""]
for title, meetings in [("All 4 meetings (no sampled)", MEETINGS), ("3 common meetings (all methods)", COMMON)]:
    t = totals(meetings)
    if t.empty:
        continue
    asr_total, gt_total = int(t["asr"].iloc[0]), int(t["gt"].iloc[0])
    md += [f"## {title}: {', '.join(meetings)}", "",
           f"ASR transcript: {asr_total}/{10 * len(meetings)}; GT transcript: {gt_total}/{10 * len(meetings)}.", "",
           "### Gain vs ASR (points)", "", wide(t, "gain vs ASR").to_markdown(), "",
           "### Questions fixed that the GT transcript also gets right / questions broken", "",
           (wide(t, "fixed (GT-fixable)").astype(str) + " / " + wide(t, "broken").astype(str)).to_markdown(), ""]

md += ["## Per meeting", "", df.drop(columns=["asr", "gt"]).to_markdown(index=False), ""]
out = "results/sampled_deltas/method_comparison.md"
with open(out, "w") as f:
    f.write("\n".join(md))
print("\n".join(md))
