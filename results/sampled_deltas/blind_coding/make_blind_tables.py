"""
Build blinded coding packets from one meeting's four top-20 tables. Each metric's table gets a random
letter (A-D); rows are shuffled and stripped of rank, line number, z and baseline correctness, so a
coder sees only (row id, question, ASR text, GT text). The letter -> metric key and row id -> line map
go to key.json, which is never shown to coders. The full coder prompt for each packet is written to
prompt_<letter>.txt (same wording for every meeting).

Run from the repo root (after top20_tables_all_metrics.py MEETING has been run):
    python results/sampled_deltas/blind_coding/make_blind_tables.py MEETING
Writes results/sampled_deltas/blind_coding/MEETING/{packet_*.md, prompt_*.txt, quiz.md, key.json}.
"""

import os
import re
import sys
import json
import random
import zlib

M = sys.argv[1] if len(sys.argv) > 1 else "ES2005d"
SRC = f"results/sampled_deltas/top20_tables_{M}.md"
OUT = f"results/sampled_deltas/blind_coding/{M}"
SEED = 20261021 if M == "ES2005d" else 20261021 + zlib.crc32(M.encode())  # keeps ES2005d's original packets

PROMPT = """You are helping with a qualitative coding task for a research project on speech recognition.
Do not use any tools: everything you need is in this message. Work only from the text below.

## Background
A recorded meeting was transcribed by an automatic speech recognizer (ASR). Separately, humans
produced a ground-truth (GT) transcript. Each row in the table below is one short line from the
meeting: the ASR text and the GT text for the same stretch of audio. Each row is paired with one
question from a 10-question comprehension quiz about the meeting. The quiz and its gold answers are:

{quiz}
## Rows to code
{packet}
## What to judge
For every row, make two independent judgments.

1. mistranscribed: Does the ASR text differ from the GT text in a way that changes or loses meaning?
   - Yes: words or content differ in a way that changes what was said (wrong words, missing or
     invented content, a different utterance altogether).
   - No: same meaning. Ignore differences only in punctuation, capitalization, number formatting
     ("7" vs "Seven"), filler words (um, uh), false starts, or repeated words.

2. important: Does the GT text help answer the paired question correctly? Judge the GT text on its
   own, regardless of whether the ASR got it right.
   - Yes: the GT text states or directly supports (part of) the gold answer.
   - Partly: the GT text gives context the question is about, but not the answer itself.
   - No: the GT text is unrelated to the question or its answer.

You only see one line at a time, without surrounding conversation. Judge each line as written;
do not guess at what nearby lines might say. Rows are in no particular order.

## Output
Reply with only a JSON array, one object per row, in the same order as the table:
[{{"row": "{letter}01", "mistranscribed": "Yes|No", "important": "Yes|Partly|No", "note": "<= 12 words"}}]"""

os.makedirs(OUT, exist_ok=True)
with open(SRC) as f:
    text = f.read()

quiz_rows = re.findall(r"^\| (q\d) \| (.+?) \| (.+?) \| (?:Yes|No) \| (?:Yes|No) \|$", text, flags=re.M)
quiz_md = "| q | question | gold answer |\n|---|---|---|\n" + "\n".join(f"| {q} | {question} | {answer} |" for q, question, answer in quiz_rows) + "\n"
sections = re.split(r"^## ", text, flags=re.M)[1:]
tables = {}
for sec in sections:
    title = sec.splitlines()[0].strip()
    rows = re.findall(r"^\| (\d+) \| (\d+) \| (.*?) \| (.*?) \| [-\d.]+ \| (q\d) \|", sec, flags=re.M)
    if rows:
        tables[title] = [{"rank": int(r), "line": int(l), "asr": a, "gt": g, "q": q} for r, l, a, g, q in rows]

rng = random.Random(SEED)
titles = list(tables)
letters = ["A", "B", "C", "D"]
rng.shuffle(titles)
key = {"meeting": M, "letters": {}, "rows": {}}
for letter, title in zip(letters, titles):
    rows = tables[title][:]
    rng.shuffle(rows)
    key["letters"][letter] = title
    md = ["| row | question | ASR text | GT text |", "|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        rid = f"{letter}{i:02d}"
        key["rows"][rid] = {"line": r["line"], "q": r["q"], "rank": r["rank"]}
        md.append(f"| {rid} | {r['q']} | {r['asr']} | {r['gt']} |")
    packet = "\n".join(md) + "\n"
    with open(f"{OUT}/packet_{letter}.md", "w") as f:
        f.write(packet)
    with open(f"{OUT}/prompt_{letter}.txt", "w") as f:
        f.write(PROMPT.format(quiz=quiz_md, packet=packet, letter=letter))

with open(f"{OUT}/quiz.md", "w") as f:
    f.write(quiz_md)
with open(f"{OUT}/key.json", "w") as f:
    json.dump(key, f, indent=2)
print(M, {k: len(v) for k, v in tables.items()}, len(quiz_rows), "quiz questions")
