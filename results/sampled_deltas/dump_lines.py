"""
Print full ASR and GT text for given ES2005d line indices (for manually judging clarifications).

    python results/sampled_deltas/dump_lines.py 3 12 13 ...
"""

import sys

AMI = "shared/datasets/amicorpus/train/ES2005d/transcripts"
with open(f"{AMI}/custom_transcript_gt_segments.txt") as f:
    asr = [l.strip() for l in f]
with open(f"{AMI}/parsed_diarized_gt.txt") as f:
    gt = [l.strip() for l in f]

for j in map(int, sys.argv[1:]):
    print(f"{j}\n  ASR: {asr[j]}\n  GT : {gt[j]}")
