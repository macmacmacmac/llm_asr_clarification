"""
ETA for the full train logp job (2400338): fits time ~ a * chars^2 on the meetings already finished
(from logp_full_train.out) and applies it to the meetings not yet done.

Run from the repo root:
    python results/sampled_deltas/logp_train_eta.py
"""

import os
import re

AMI = "shared/datasets/amicorpus/train"
TOO_LONG = {"EN2009d", "EN2001a", "EN2005a", "IN1013", "IN1016", "EN2006a", "TS3006d"}

with open("logp_full_train.out", errors="ignore") as f:
    log = f.read()
done = {m: float(s) for m, s in re.findall(r"\[(\w+)\] Log probs generation took ([\d.]+) seconds", log)}
prompt = {m: float(s) for m, s in re.findall(r"\[(\w+)\] Prompt generation took ([\d.]+) seconds", log)}

chars = {m: os.path.getsize(f"{AMI}/{m}/transcripts/custom_transcript_gt_segments.txt")
         for m in sorted(os.listdir(AMI)) if m not in TOO_LONG}
total = {m: done[m] + prompt.get(m, 0) for m in done}
a = sum(total.values()) / sum(chars[m] ** 2 for m in total)
remaining = [m for m in chars if m not in done]
eta_h = a * sum(chars[m] ** 2 for m in remaining) / 3600

print(f"done {len(done)}/{len(chars)} meetings ({sum(total.values()) / 3600:.1f} h of log-prob + prompt time)")
print(f"remaining {len(remaining)} meetings ({remaining[0]} .. {remaining[-1]}): ~{eta_h:.1f} h")
