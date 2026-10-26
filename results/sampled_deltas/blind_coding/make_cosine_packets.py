"""
Add the cosine importance detector (pipelinev4's GTImportanceDetector, "GT") to the blind coding as
packet E for each meeting.

Ranking: exactly as GTImportanceDetector — all-MiniLM-L6-v2 cosine similarity between each GT line
(text after the last ':') and the gold answers of quiz_from_parsed_diarized_gt.json (the older GPT
quiz); line score = max over answers. Top 20 are checked against the lines that pipelinev4 actually
replaced in custom_transcript_gt_segments_all_gt_20_clarify4.txt.

Question pairing (needed by coders; the old quiz's questions differ from vllm_quiz.json): each line is
paired with the vllm_quiz.json question whose gold answer has the highest cosine similarity to the line.

Rows are shuffled, the prompt is identical to the other packets (built from prompt_A.txt), and the
mapping is appended to key.json under letter E.

Run from the repo root (llm_clarification env, CPU is fine):
    python results/sampled_deltas/blind_coding/make_cosine_packets.py
"""

import re
import json
import random

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

AMI = "shared/datasets/amicorpus/train"
D = "results/sampled_deltas/blind_coding"
MEETINGS = ["ES2005d", "IN1012", "TS3005d"]
SEED = 47
TITLE = "Cosine (GT importance detector)"

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

for m in MEETINGS:
    with open(f"{AMI}/{m}/transcripts/parsed_diarized_gt.txt") as f:
        gt_lines = f.read().strip().split("\n")
    with open(f"{AMI}/{m}/transcripts/custom_transcript_gt_segments.txt") as f:
        asr_lines = f.read().strip().split("\n")
    with open(f"{AMI}/{m}/quiz/quiz_from_parsed_diarized_gt.json") as f:
        old_answers = [q["correct_answer"] for q in json.load(f)]
    with open(f"{AMI}/{m}/quiz/vllm_quiz.json") as f:
        new_answers = [q["correct_answer"] for q in json.load(f)]

    seg_emb = model.encode([s.split(":")[-1] for s in gt_lines], convert_to_tensor=True)
    score = cos_sim(model.encode(old_answers, convert_to_tensor=True), seg_emb).max(dim=0).values
    order = sorted(range(len(gt_lines)), key=lambda i: float(score[i]), reverse=True)[:20]
    pair_q = cos_sim(seg_emb, model.encode(new_answers, convert_to_tensor=True)).argmax(dim=1)

    with open(f"{AMI}/{m}/transcripts/custom_transcript_gt_segments_all_gt_20_clarify4.txt") as f:
        clarified = f.read().strip().split("\n")
    changed = {i for i, (a, c) in enumerate(zip(asr_lines, clarified)) if a != c}
    expected = {i for i in order if asr_lines[i] != gt_lines[i]}
    print(f"{m}: top-20 lines that differ from ASR = {len(expected)}; pipelinev4 changed {len(changed)}; match = {expected == changed}")

    def clean(line):
        return line.split("]: ", 1)[-1].replace("|", "/")

    rows = [{"rank": r, "line": j, "q": f"q{int(pair_q[j])}"} for r, j in enumerate(order, 1)]
    random.Random(SEED + len(m)).shuffle(rows)
    packet = ["| row | question | ASR text | GT text |", "|---|---|---|---|"]
    with open(f"{D}/{m}/key.json") as f:
        key = json.load(f)
    key["letters"]["E"] = TITLE
    for i, r in enumerate(rows, 1):
        rid = f"E{i:02d}"
        key["rows"][rid] = r
        packet.append(f"| {rid} | {r['q']} | {clean(asr_lines[r['line']])} | {clean(gt_lines[r['line']])} |")
    packet_md = "\n".join(packet) + "\n"

    with open(f"{D}/{m}/prompt_A.txt") as f:
        template = f.read()
    prompt = re.sub(r"## Rows to code\n.*?\n## What to judge", lambda _: f"## Rows to code\n{packet_md}\n## What to judge",
                    template, flags=re.S).replace('"row": "A01"', '"row": "E01"')

    with open(f"{D}/{m}/packet_E.md", "w") as f:
        f.write(packet_md)
    with open(f"{D}/{m}/prompt_E.txt", "w") as f:
        f.write(prompt)
    with open(f"{D}/{m}/key.json", "w") as f:
        json.dump(key, f, indent=2)
