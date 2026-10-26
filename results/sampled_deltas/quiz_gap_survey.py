"""
Ad hoc survey: how much room is there for clarification to help? Per meeting, the official
Qwen3-32B quiz scores (vllm_quiz.json) on the ASR transcript vs the GT transcript, and the number of
"fixable" questions (ASR wrong, GT right). Also checks which meetings have teacher labels.

Run from the repo root:
    python results/sampled_deltas/quiz_gap_survey.py
"""

import os
import json

import torch
import pandas as pd

AMI = "shared/datasets/amicorpus"
ASR = "score_using_custom_transcript_gt_segments"
GT = "score_using_parsed_diarized_gt"

logp8 = set(torch.load("shared/datasets/logp_deltas/train_Qwen3-8B-FP8.pt"))
logp32 = set(torch.load("shared/datasets/logp_deltas/train_32b.pt"))
sampled = {f[:-3] for f in os.listdir("shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7") if f.endswith(".pt")}
print("logp 8B meetings:", sorted(logp8))
print("logp 32B meetings:", sorted(logp32))
print("sampled meetings:", sorted(sampled))

rows = []
for split in ["train", "validation", "test"]:
    for m in sorted(os.listdir(f"{AMI}/{split}")):
        path = f"{AMI}/{split}/{m}/quiz/vllm_quiz.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            quiz = json.load(f)
        if not quiz or not all(ASR in q and GT in q for q in quiz):
            continue
        asr = sum(q[ASR] for q in quiz)
        gt = sum(q[GT] for q in quiz)
        fixable = sum((not q[ASR]) and q[GT] for q in quiz)
        rows.append({"split": split, "meeting": m, "asr": asr, "gt": gt, "gap": gt - asr, "fixable": fixable})

df = pd.DataFrame(rows)
print(f"\n{len(df)} meetings with both scores")
print(df.groupby("split")[["asr", "gt", "gap", "fixable"]].agg(["mean", "sum", "count"]).round(2).to_string())
print("\nFixable-question distribution (train):", df[df.split == "train"].fixable.value_counts().sort_index().to_dict())
print("\nPilot meetings:")
print(df[df.meeting.isin(["ES2005d", "IN1012", "TS3005d", "IS1007c"])].to_string(index=False))
print("\nTrain meetings with >= 3 fixable questions:", df[(df.split == "train") & (df.fixable >= 3)].meeting.tolist())
