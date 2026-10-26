"""
Ad hoc pilot check: for the questions the sampled-8B top-20 concentrates on, compare the official
quiz run's baseline answer/score (vllm_quiz.json) with the sampled-8B base-transcript answers.

Run from the repo root:
    python results/sampled_deltas/baseline_mismatch.py
"""

import json
from collections import Counter

import torch

R = "shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7"
AMI = "shared/datasets/amicorpus/train"
CASES = [("ES2005d", 4), ("IN1012", 1), ("TS3005d", 9), ("TS3005d", 3)]

for m, q in CASES:
    d = torch.load(f"{R}/{m}.pt", weights_only=False)
    with open(f"{AMI}/{m}/quiz/vllm_quiz.json") as f:
        quiz = json.load(f)
    tab = d["answer_table"]
    base_ids = d["answer_ids"][0, q].tolist()
    p_lines = d["p_correct"][1:, q]
    print(f"== {m} q{q}: {quiz[q]['question']}")
    print(f"   gold:     {quiz[q]['correct_answer']}")
    print(f"   official: score={quiz[q]['score_using_custom_transcript_gt_segments']} | {quiz[q]['answer_using_custom_transcript_gt_segments'][:160]}")
    print(f"   sampled base p={d['p_correct'][0, q]:.2f}; mean p over lines={p_lines.mean():.2f}; frac lines p>=0.5: {(p_lines >= 0.5).float().mean():.2f}")
    for i, n in Counter(base_ids).most_common(3):
        print(f"     x{n} score={tab[i]['score']} | {tab[i]['answer'][:160]}")
