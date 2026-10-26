"""
Extrapolate the GPU-hours of a full Qwen3-32B logp-delta run from the EN2001e calibration job
(job 2399897, RTX6000B): 238 s prompt building + 7002 s log probs for EN2001e (101,157 chars).
Per-meeting cost is assumed to scale with (transcript chars)^2 (versions x prompt length), plus a
fixed ~5 min model load per job. Excludes the 7 meetings generate_logp_deltas.py treats as too long.

Run from the repo root:
    python results/sampled_deltas/logp_cost_estimate.py
"""

import os

AMI = "shared/datasets/amicorpus"
TOO_LONG = {"EN2009d", "EN2001a", "EN2005a", "IN1013", "IN1016", "EN2006a", "TS3006d"}
PILOT = {"ES2005d", "IN1012", "IS1007c", "TS3005d"}
REF_CHARS, REF_SEC = 101157, 238 + 7002
LOAD_H = 5 / 60

for split in ["train", "validation"]:
    rows = []
    for m in sorted(os.listdir(f"{AMI}/{split}")):
        path = f"{AMI}/{split}/{m}/transcripts/custom_transcript_gt_segments.txt"
        if m in TOO_LONG or not os.path.exists(path):
            continue
        hours = REF_SEC * (os.path.getsize(path) / REF_CHARS) ** 2 / 3600
        rows.append((m, hours))
    remaining = [h for m, h in rows if m not in PILOT and m != "EN2001e"]
    pilot = [h for m, h in rows if m in PILOT]
    print(f"{split}: {len(rows)} meetings total; remaining {len(remaining)} = {sum(remaining):.1f} GPU-h "
          f"(largest {max(remaining):.2f} h, median {sorted(remaining)[len(remaining) // 2]:.2f} h); "
          f"pilot {len(pilot)} = {sum(pilot):.1f} GPU-h")
print(f"+ ~{LOAD_H * 60:.0f} min model load per job")
