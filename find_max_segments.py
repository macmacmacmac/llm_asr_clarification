#!/usr/bin/env python3
"""Script to find the meeting transcript with the greatest number of segments/lines."""

from pathlib import Path


def find_meeting_with_max_segments(train_dir: Path) -> tuple[str, int, Path]:
    max_lines = -1
    max_meeting = ""
    max_file = None

    # Search for all parsed_diarized_gt.txt files within the train split
    for transcript_path in train_dir.glob("*/transcripts/parsed_diarized_gt.txt"):
        meeting_name = transcript_path.parent.parent.name
        with open(transcript_path, "r", encoding="utf-8") as f:
            # Count non-empty lines (or total lines)
            lines = [line for line in f if line.strip()]
            num_lines = len(lines)

        if num_lines > max_lines:
            max_lines = num_lines
            max_meeting = meeting_name
            max_file = transcript_path

    return max_meeting, max_lines, max_file


def main():
    train_dir = Path("/home/pkongsomjit/Projects/llm_asr_clarification/shared/datasets/amicorpus/train")
    meeting, count, path = find_meeting_with_max_segments(train_dir)
    print(f"Meeting with greatest number of segments: {meeting}")
    print(f"Number of segments/lines: {count}")
    print(f"Transcript path: {path}")


if __name__ == "__main__":
    main()
