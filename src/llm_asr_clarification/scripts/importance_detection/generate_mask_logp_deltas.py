"""
Generate teacher-forced quiz answer log-probability drops from masking each GT segment.

Masking Idea
------------
generate_logp_deltas.py starts from the ASR transcript and swaps in one GT line at a time, so its label
is high only if a line is both important AND mistranscribed. Here we start from the clean GT transcript
and mask one line at a time instead. The drop in the quiz answers' log-probs measures how much the quiz
depends on that line, i.e. how much it would hurt if that line were mistranscribed — independent of how
well the ASR actually did on it.

- Version 0 is the full GT transcript; version j+1 has only line j masked.
- `--mask-mode placeholder` keeps the "(start - end)[Speaker X]: " prefix and replaces the text with
  `--mask-token`; `--mask-mode delete` removes the line entirely.
- Label: drop vs the full GT, divided by the per-question std (floored at `--tau`), clamped at 0, max
  over questions. 0 means "masking this line changed nothing".

Determinism Notes
-----------------
Same settings as generate_logp_deltas.py (see its docstring for why):
1. `VLLM_ENABLE_V1_MULTIPROCESSING="0"` — runs the vLLM engine synchronously in one process.
2. `enforce_eager=True` in `LLM(...)` — disables `torch.compile` and CUDA graphs.
3. `async_scheduling=False` in `LLM(...)` — keeps scheduler batching deterministic.
4. `seed=47` and `temperature=0.0` in `LLM(...)` / `SamplingParams(...)` — fixes RNG state.
5. `llm.reset_prefix_cache()` before each meeting — no KV cache hits across meetings.
6. Deterministic prompt deduplication and fixed `chunk_size = 1000` per meeting.
"""

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import re
import json
import gc
import time
import argparse
import torch
from tqdm.auto import tqdm
from llm_asr_clarification import get_logger
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.distributed.parallel_state import destroy_model_parallel


# PROMPTS
SYSTEM_PROMPT = """You are an expert at paying attention to meetings and answering quiz questions about them.
You will be shown a transcription from a meeting and a single quiz question.
Your task is to answer the question based only on the information in the transcript.
Respond with only the answer as plain text. Do not include any preamble or extra text."""

USER_PROMPT = """## Transcript:
{transcript}

## Question:
{question}"""

PREFIX_RE = re.compile(r"^\(.*?\)\[.*?\]:\s*")


def mask_line(line, mask_token):
    prefix = PREFIX_RE.match(line)
    prefix = prefix.group(0) if prefix else ""
    return f"{prefix}{mask_token}"


def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--ami-path", type=str, default="./shared/datasets/amicorpus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--output-name", type=str, default="run1")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--meetings", type=str, nargs="+", default=None, help="Only process these meetings (e.g. ES2005d)")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--mask-mode", type=str, choices=["placeholder", "delete"], default="placeholder")
    parser.add_argument("--mask-token", type=str, default="[inaudible]")
    parser.add_argument("--tau", type=float, default=0.05, help="Floor (nats) for the per-question std scale")

    args, _ = parser.parse_known_args(args_list)
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"--shard-id ({args.shard_id}) must be in [0, {args.num_shards})")

    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    logger.info(f"Loading LLM ({args.model})...")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=47,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        async_scheduling=False,
        enforce_eager=True,
    )

    tokenizer = llm.get_tokenizer()

    # max_tokens=1 because vLLM must generate at least one token; that token is ignored.
    # We only read prompt_logprobs, i.e. teacher-forced log probs of the prompt (incl. the answer) tokens.
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=47,
        prompt_logprobs=1,
        max_tokens=1,
        skip_reading_prefix_cache=False,  # Force prefix caching even with prompt_logprobs
    )


    logp_deltas = {}
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    save_path = f"./shared/datasets/logp_mask_deltas/{args.split}_{args.output_name}{shard_suffix}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        logp_deltas = torch.load(save_path)
        logger.info(f"Loaded existing logp deltas for {len(logp_deltas)} meetings.")

    split_path = os.path.join(args.ami_path, args.split)
    if not os.path.exists(split_path):
        logger.error(f"Unable to find folder: {split_path}")
        return

    # Filtering meetings
    meetings_too_long = {
        "EN2009d",
        "EN2001a",
        "EN2005a",
        "IN1013",
        "IN1016",
        "EN2006a",
        "TS3006d"
    }

    if args.meetings:
        target_filter = lambda name: name in set(args.meetings)
    else:
        target_filter = lambda name: name not in meetings_too_long
    meeting_paths = sorted(entry.path for entry in os.scandir(split_path) if entry.is_dir() and target_filter(entry.name))
    meeting_paths = meeting_paths[args.shard_id :: args.num_shards]
    if args.reverse:
        meeting_paths = list(reversed(meeting_paths))
    logger.info(
        f"Shard {args.shard_id}/{args.num_shards} (reverse={args.reverse}): "
        f"assigned {len(meeting_paths)} meetings {[os.path.basename(p) for p in meeting_paths]} (saving to {save_path})."
    )

    for meeting_path in tqdm(meeting_paths, desc="Processing Meetings", position=0, leave=True):
        meeting_name = os.path.basename(meeting_path)

        if meeting_name in logp_deltas:
            logger.info(f"Skipping {meeting_name}, already processed.")
            continue

        gt_path = os.path.join(meeting_path, "transcripts", "parsed_diarized_gt.txt")
        quiz_path = os.path.join(meeting_path, "quiz", "vllm_quiz.json")

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_lines = [line.strip() for line in f]
        except FileNotFoundError as e:
            logger.warning(f"Unable to find GT transcript for {meeting_name}: {e}")
            continue

        try:
            with open(quiz_path, "r", encoding="utf-8") as f:
                quiz = json.loads(f.read())
        except FileNotFoundError:
            logger.warning(f"Unable to find the quiz at {quiz_path}")
            continue
        except json.JSONDecodeError as e:
            logger.warning(f"JSONDecodeError reading {quiz_path}: {e}")
            continue

        if len(quiz) == 0:
            logger.warning(f"[{meeting_name}] Quiz is empty!")
            continue

        # Build Transcript Versions (Non-Cumulative)
        # Version 0 is the full GT transcript; version j+1 has only line j masked.
        transcript_versions = ["\n".join(gt_lines)]
        for j in range(len(gt_lines)):
            gt_copy = gt_lines.copy()
            if args.mask_mode == "placeholder":
                gt_copy[j] = mask_line(gt_lines[j], args.mask_token)
            else:
                del gt_copy[j]
            transcript_versions.append("\n".join(gt_copy))


        # Logp calculation phase
        logger.info(f"[{meeting_name}] Building log prob calculation prompts for {len(transcript_versions)} versions...")
        start_time = time.time()
        answer_ids_per_q = [tokenizer(q["correct_answer"], add_special_tokens=False)["input_ids"] for q in quiz]

        prompt_to_unique_idx = {}
        unique_prompt_texts = []
        unique_answer_ids = []

        # Keep track of which unique prompt belongs to which (version, question) pair
        # unique_idx_map[v_idx][q_idx] = unique_idx
        unique_idx_map = [[0] * len(quiz) for _ in range(len(transcript_versions))]

        for v_idx, transcript_text in enumerate(transcript_versions):
            for q_idx, (q, answer_ids) in enumerate(zip(quiz, answer_ids_per_q)):
                prompt_text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT.format(
                            transcript=transcript_text,
                            question=q["question"]
                        )}
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                # If we haven't seen this exact prompt text before, assign it a new ID
                if prompt_text not in prompt_to_unique_idx:
                    prompt_to_unique_idx[prompt_text] = len(unique_prompt_texts)
                    unique_prompt_texts.append(prompt_text)
                    unique_answer_ids.append(answer_ids)

                # Record the ID assigned to this specific (version, question)
                unique_idx_map[v_idx][q_idx] = prompt_to_unique_idx[prompt_text]

        logger.info(f"[{meeting_name}] Tokenizing {len(unique_prompt_texts)} unique prompts in parallel (out of {len(transcript_versions) * len(quiz)} total)...")
        prefix_ids_list = tokenizer(unique_prompt_texts, add_special_tokens=False)["input_ids"]

        prompts = []
        answer_spans = []
        for prefix_ids, answer_ids in zip(prefix_ids_list, unique_answer_ids):
            prompt_ids = list(prefix_ids) + answer_ids
            prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
            answer_spans.append((len(prefix_ids), prompt_ids))

        logger.info(f"[{meeting_name}] Prompt generation took {time.time() - start_time:.2f} seconds.")

        max_prompt_len = max(len(p["prompt_token_ids"]) for p in prompts)
        if max_prompt_len > args.max_model_len:
            logger.warning(f"[{meeting_name}] Prompt too long ({max_prompt_len} > {args.max_model_len}), skipping.")
            continue

        llm.reset_prefix_cache()
        logger.info(f"[{meeting_name}] Generating {len(prompts)} log probs in chunks...")
        start_time = time.time()

        unique_logps = []
        chunk_size = 1000

        for chunk_start in range(0, len(prompts), chunk_size):
            chunk_prompts = prompts[chunk_start : chunk_start + chunk_size]
            chunk_spans = answer_spans[chunk_start : chunk_start + chunk_size]

            chunk_outputs = llm.generate(chunk_prompts, sampling_params)

            for i, output in enumerate(chunk_outputs):
                start, prompt_ids = chunk_spans[i]
                ans_len = len(prompt_ids) - start
                plp = output.prompt_logprobs

                plp_ans = plp[-ans_len:]
                ans_ids = prompt_ids[-ans_len:]

                sum_logp = 0.0
                for k in range(ans_len):
                    if plp_ans[k] is None:
                        continue
                    sum_logp += plp_ans[k][ans_ids[k]].logprob

                unique_logps.append(sum_logp)

        logger.info(f"[{meeting_name}] Log probs generation took {time.time() - start_time:.2f} seconds.")

        # Map the unique logprobs back to the explicit full matrix
        num_q = len(quiz)
        logps = torch.zeros(len(transcript_versions), num_q, dtype=torch.float64)

        for v_idx in range(len(transcript_versions)):
            for q_idx in range(num_q):
                u_idx = unique_idx_map[v_idx][q_idx]
                logps[v_idx, q_idx] = unique_logps[u_idx]

        del prompts, answer_spans

        # Drop vs the full GT transcript (>0 = masking line j made question q's answer less likely),
        # scaled by the per-question std (floored at tau), clamped at 0, max over questions.
        drop = logps[0:1] - logps[1:]
        std = drop.std(dim=0)
        segment_deltas = (drop / std.clamp(min=args.tau)).clamp(min=0).max(dim=1).values
        num_quiet = int((std < args.tau).sum())

        logp_deltas[meeting_name] = {
            "segment_deltas": segment_deltas,  # [N]; index j = scaled answer logp drop from masking GT line j
            "logps": logps,                    # [N+1, Q] raw summed answer log probs per question (row 0 = full GT)
        }
        torch.save(logp_deltas, save_path)

        logger.info(
            f"[{meeting_name}] Finished! N={len(gt_lines)}, quiet columns (std < tau): {num_quiet}/{num_q}, "
            f"max label={segment_deltas.max().item():.2f}. Saved delta log probs."
        )

    logger.info("Destroying LLM...")
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("All tasks completed successfully!")
