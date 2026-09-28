#!/usr/bin/env python3
"""Script to analyze meeting token lengths against max-model-len (40960)."""

import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from llm_asr_clarification.constants.quiz_prompts import (
    VLLM_ANSWER_SYSTEM_PROMPT,
    VLLM_ANSWER_USER_PROMPT,
)


def analyze_meetings(base_dir: Path, model_name: str = "Qwen/Qwen3-32B-FP8"):
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []

    splits = ["train", "validation", "test"]
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue

        for meeting_dir in sorted(split_dir.iterdir()):
            if not meeting_dir.is_dir():
                continue

            meeting_name = meeting_dir.name
            gt_path = meeting_dir / "transcripts" / "parsed_diarized_gt.txt"
            asr_path = meeting_dir / "transcripts" / "custom_transcript_gt_segments.txt"
            quiz_path = meeting_dir / "quiz" / "vllm_quiz.json"

            gt_text = gt_path.read_text(encoding="utf-8").strip() if gt_path.exists() else ""
            asr_text = asr_path.read_text(encoding="utf-8").strip() if asr_path.exists() else ""

            if not gt_text and not asr_text:
                continue

            gt_tokens = len(tokenizer(gt_text)["input_ids"]) if gt_text else 0
            asr_tokens = len(tokenizer(asr_text)["input_ids"]) if asr_text else 0

            # Quiz questions length
            max_q_tokens = 0
            sample_q = "What was discussed in the meeting?"
            if quiz_path.exists():
                try:
                    with open(quiz_path, "r", encoding="utf-8") as f:
                        quiz_data = json.load(f)
                    for item in quiz_data:
                        q_toks = len(tokenizer(item.get("question", ""))["input_ids"])
                        if q_toks > max_q_tokens:
                            max_q_tokens = q_toks
                            sample_q = item.get("question", "")
                except Exception:
                    pass

            # Full prompt length with chat template
            # Test with GT and ASR
            def get_prompt_len(transcript):
                if not transcript:
                    return 0
                messages = [
                    {"role": "system", "content": VLLM_ANSWER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": VLLM_ANSWER_USER_PROMPT.format(
                            transcript=transcript, question=sample_q
                        ),
                    },
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return len(tokenizer(text)["input_ids"])

            gt_prompt_tokens = get_prompt_len(gt_text)
            asr_prompt_tokens = get_prompt_len(asr_text)

            results.append({
                "split": split,
                "meeting": meeting_name,
                "gt_tokens": gt_tokens,
                "asr_tokens": asr_tokens,
                "gt_prompt_tokens": gt_prompt_tokens,
                "asr_prompt_tokens": asr_prompt_tokens,
                "max_prompt_tokens": max(gt_prompt_tokens, asr_prompt_tokens),
            })

    return results


def main():
    base_dir = Path("/home/pkongsomjit/Projects/llm_asr_clarification/shared/datasets/amicorpus")
    results = analyze_meetings(base_dir)

    print(f"\nTotal meetings analyzed: {len(results)}")

    limit = 40960
    buffer = 200
    thresh_tight = limit - buffer  # 40760
    thresh_loose = limit + buffer  # 41160

    for split in ["train", "validation", "test", "all"]:
        if split == "all":
            subset = results
        else:
            subset = [r for r in results if r["split"] == split]

        total = len(subset)
        gt_over_limit = [r for r in subset if r["gt_prompt_tokens"] > limit]
        asr_over_limit = [r for r in subset if r["asr_prompt_tokens"] > limit]
        either_over_limit = [r for r in subset if r["max_prompt_tokens"] > limit]

        gt_over_tight = [r for r in subset if r["gt_prompt_tokens"] > thresh_tight]
        either_over_tight = [r for r in subset if r["max_prompt_tokens"] > thresh_tight]

        print(f"\n{'='*30} Split: {split.upper()} (Total: {total}) {'='*30}")
        print(f"Meetings exceeding {limit} (Prompt with max(GT, ASR)): {len(either_over_limit)} / {total}")
        for r in either_over_limit:
            print(f"  - {r['meeting']}: GT prompt={r['gt_prompt_tokens']}, ASR prompt={r['asr_prompt_tokens']}")

        print(f"Meetings exceeding {thresh_tight} ({limit} - {buffer} buffer): {len(either_over_tight)} / {total}")
        for r in either_over_tight:
            if r not in either_over_limit:
                print(f"  - {r['meeting']} (in buffer zone): GT prompt={r['gt_prompt_tokens']}, ASR prompt={r['asr_prompt_tokens']}")

    # Also show top 10 longest meetings overall
    print("\n" + "="*70)
    print("TOP 10 LONGEST MEETINGS OVERALL:")
    sorted_res = sorted(results, key=lambda x: x["max_prompt_tokens"], reverse=True)
    for i, r in enumerate(sorted_res[:10], 1):
        print(f"{i:2d}. [{r['split']}] {r['meeting']}: Max Prompt Tokens = {r['max_prompt_tokens']} (GT={r['gt_prompt_tokens']}, ASR={r['asr_prompt_tokens']})")


if __name__ == "__main__":
    main()
