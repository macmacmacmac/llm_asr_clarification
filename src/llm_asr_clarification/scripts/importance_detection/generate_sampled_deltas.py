"""
    Sampled-correctness counterpart of generate_logp_deltas.py.

    For every meeting, version 0 is the base ASR transcript and version j+1 has only line j
    replaced by its GT counterpart (window 1, non-cumulative), exactly as in generate_logp_deltas.py.
    For every (version, question) we sample K answers at temperature > 0 with the same plain-text
    prompts as the logp script, then grade each distinct answer once with the VLLM_SCORER prompts
    at temperature 0. The result is an estimate of p(correct | version, question) per line.

    Determinism is not a goal here, so vLLM is left to batch freely. Speed comes from:
      - n=K parallel sampling, so the K samples of a prompt share one prefill;
      - prefix caching + version-ordered prompts: version j shares lines 0..j-1 with the base,
        and all questions of a version share the whole transcript;
      - versions whose replaced line is identical to the ASR line are not re-sampled (they are the
        base prompt verbatim) and reuse the base samples;
      - the judge only sees each distinct (question, normalized answer) once.

    Seeding: every request uses the same seed (sample k gets seed + k), i.e. common random numbers
    across versions. Each version's K samples are still proper independent draws, but two versions
    with near-identical logits tend to draw the same answers, which lowers the variance of the
    version-vs-version differences we care about. Pass --seed -1 for unseeded sampling.

    Output: one file per meeting at <out-dir>/<split>_<model>_K<K>_T<temperature>/<meeting>.pt
    so that shards running in parallel never write the same file. Labels are derived later on CPU
    by analyze_sampled_deltas.py; this script only stores raw outcomes.
"""

import os
import re
import gc
import json
import time
import argparse

import torch
from tqdm.auto import tqdm

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.sampling_params import StructuredOutputsParams

from llm_asr_clarification import get_logger
from llm_asr_clarification.constants.quiz_prompts import (
    VLLM_SCORER_SYSTEM_PROMPT, VLLM_SCORER_USER_PROMPT
)
from pydantic import BaseModel
from typing import Literal


# PROMPTS (identical to generate_logp_deltas.py, so answers are comparable with the logp deltas)
SYSTEM_PROMPT = """You are an expert at paying attention to meetings and answering quiz questions about them.
You will be shown a transcription from a meeting and a single quiz question.
Your task is to answer the question based only on the information in the transcript.
Respond with only the answer as plain text. Do not include any preamble or extra text."""

USER_PROMPT = """## Transcript:
{transcript}

## Question:
{question}"""


class ScoreResponse(BaseModel):
    score: Literal[0, 1]


MEETINGS_TOO_LONG = {
    "EN2009d",
    "EN2001a",
    "EN2005a",
    "IN1013",
    "IN1016",
    "EN2006a",
    "TS3006d"
}

THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def normalize_answer(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = " ".join(text.lower().split())
    return text.strip(" .")


def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-FP8")
    parser.add_argument("--ami-path", type=str, default="./shared/datasets/amicorpus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--meetings", type=str, nargs="+", default=None, help="Explicit meetings to run (default: all in split minus the too-long ones)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-samples", "-k", type=int, default=16, help="Answers sampled per (version, question)")
    # Qwen3 recommended non-thinking sampling settings
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=47, help="Shared per-request seed (common random numbers); -1 for unseeded")
    parser.add_argument("--versions-per-chunk", type=int, default=50, help="Versions submitted per llm.generate call; bounds host memory for tokenized prompts")
    parser.add_argument("--out-dir", type=str, default="./shared/datasets/sampled_deltas")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--kv-cache-dtype", type=str, default="auto", help="'fp8' halves KV memory/bandwidth (decode is KV-bandwidth bound here)")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Leave None for vLLM's default")

    args, _ = parser.parse_known_args(args_list)

    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    split_path = os.path.join(args.ami_path, args.split)
    if not os.path.exists(split_path):
        logger.error(f"Unable to find folder: {split_path}")
        return

    model_name = args.model.split("/")[-1]
    save_dir = os.path.join(args.out_dir, f"{args.split}_{model_name}_K{args.num_samples}_T{args.temperature}")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving per-meeting results to {save_dir}")

    if args.meetings is not None:
        meeting_names = sorted(args.meetings)
    else:
        meeting_names = sorted(entry.name for entry in os.scandir(split_path) if entry.is_dir() and entry.name not in MEETINGS_TOO_LONG)
    meeting_names = meeting_names[args.shard_idx::args.num_shards]

    # Skip finished meetings before paying for model load
    meeting_names = [m for m in meeting_names if not os.path.exists(os.path.join(save_dir, f"{m}.pt"))]
    logger.info(f"Meetings to process ({len(meeting_names)}): {meeting_names}")
    if not meeting_names:
        logger.info("Nothing to do.")
        return

    logger.info(f"Loading LLM ({args.model})...")
    llm_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=47,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    seed = None if args.seed < 0 else args.seed
    answer_sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_answer_tokens,
        seed=seed,
    )
    score_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        structured_outputs=StructuredOutputsParams(json=json.dumps(ScoreResponse.model_json_schema()))
    )

    def build_prompt(transcript_text, question):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(transcript=transcript_text, question=question)}
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    for meeting_name in tqdm(meeting_names, desc="Processing Meetings", position=0, leave=True):
        meeting_path = os.path.join(split_path, meeting_name)
        save_path = os.path.join(save_dir, f"{meeting_name}.pt")

        asr_path = os.path.join(meeting_path, "transcripts", "custom_transcript_gt_segments.txt")
        gt_path = os.path.join(meeting_path, "transcripts", "parsed_diarized_gt.txt")
        quiz_path = os.path.join(meeting_path, "quiz", "vllm_quiz.json")

        try:
            with open(asr_path, "r", encoding="utf-8") as f:
                asr_lines = [line.strip() for line in f]
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_lines = [line.strip() for line in f]
        except FileNotFoundError as e:
            logger.warning(f"Unable to find transcripts for {meeting_name}: {e}")
            continue

        if len(asr_lines) != len(gt_lines):
            logger.warning(f"Line count mismatch for {meeting_name}: ASR {len(asr_lines)} != GT {len(gt_lines)}")
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

        num_lines, num_q, K = len(asr_lines), len(quiz), args.num_samples
        num_versions = num_lines + 1
        meeting_start = time.time()

        # Version 0 is the base ASR transcript; version j+1 has only line j replaced by GT.
        # A version whose replaced line equals the ASR line is the base prompt verbatim: don't sample it.
        is_noop = [asr_lines[j] == gt_lines[j] for j in range(num_lines)]
        versions_to_sample = [0] + [j + 1 for j in range(num_lines) if not is_noop[j]]
        logger.info(f"[{meeting_name}] {num_lines} lines, {num_q} questions, {sum(is_noop)} no-op lines; sampling {len(versions_to_sample)} versions x {num_q} questions x {K} samples")

        base_prompt_len = len(tokenizer(build_prompt("\n".join(asr_lines), quiz[0]["question"]), add_special_tokens=False)["input_ids"])
        if base_prompt_len + args.max_answer_tokens > args.max_model_len:
            logger.warning(f"[{meeting_name}] Prompt too long ({base_prompt_len} + {args.max_answer_tokens} > {args.max_model_len}), skipping.")
            continue

        # Answering phase. answer_texts[v][q] = list of K raw answers.
        answer_texts = {}
        num_truncated = 0
        num_prompt_tokens, num_cached_tokens = 0, 0
        answer_start = time.time()
        for chunk_start in range(0, len(versions_to_sample), args.versions_per_chunk):
            chunk_versions = versions_to_sample[chunk_start : chunk_start + args.versions_per_chunk]

            # Version-major, question-minor order keeps prompts that share a prefix adjacent
            prompts = []
            for v_idx in chunk_versions:
                lines = list(asr_lines)
                if v_idx > 0:
                    lines[v_idx - 1] = gt_lines[v_idx - 1]
                transcript_text = "\n".join(lines)
                for q in quiz:
                    prompts.append(build_prompt(transcript_text, q["question"]))

            outputs = llm.generate(prompts, answer_sampling_params, use_tqdm=False)

            for i, output in enumerate(outputs):
                v_idx = chunk_versions[i // num_q]
                q_idx = i % num_q
                answer_texts.setdefault(v_idx, [None] * num_q)[q_idx] = [c.text for c in output.outputs]
                num_truncated += sum(c.finish_reason == "length" for c in output.outputs)
                num_prompt_tokens += len(output.prompt_token_ids)
                num_cached_tokens += output.num_cached_tokens or 0

            logger.info(f"[{meeting_name}] Answered {min(chunk_start + args.versions_per_chunk, len(versions_to_sample))}/{len(versions_to_sample)} versions "
                        f"({time.time() - answer_start:.0f}s elapsed, prefix cache hit {num_cached_tokens / max(num_prompt_tokens, 1):.1%})")
            del prompts, outputs

        answer_duration = time.time() - answer_start

        # Scoring phase: grade each distinct (question, normalized answer) once.
        answer_table = []  # list of {"q_idx", "answer"} (first raw form seen)
        answer_key_to_id = {}
        answer_ids = torch.zeros(num_versions, num_q, K, dtype=torch.int32)
        for v_idx in range(num_versions):
            # No-op versions reuse the base samples
            src_v = v_idx if v_idx in answer_texts else 0
            for q_idx in range(num_q):
                for k, text in enumerate(answer_texts[src_v][q_idx]):
                    key = (q_idx, normalize_answer(text))
                    if key not in answer_key_to_id:
                        answer_key_to_id[key] = len(answer_table)
                        answer_table.append({"q_idx": q_idx, "answer": THINK_RE.sub("", text).strip()})
                    answer_ids[v_idx, q_idx, k] = answer_key_to_id[key]

        logger.info(f"[{meeting_name}] Scoring {len(answer_table)} distinct answers (out of {len(answer_texts) * num_q * K} sampled)...")
        score_start = time.time()
        score_messages = [
            [
                {"role": "system", "content": VLLM_SCORER_SYSTEM_PROMPT},
                {"role": "user", "content": VLLM_SCORER_USER_PROMPT.format(
                    question=quiz[entry["q_idx"]]["question"],
                    correct_answer=quiz[entry["q_idx"]]["correct_answer"],
                    predicted_answer=entry["answer"]
                )}
            ]
            for entry in answer_table
        ]
        score_outputs = llm.chat(
            messages=score_messages,
            sampling_params=score_sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        num_score_parse_failures = 0
        answer_scores = torch.zeros(len(answer_table), dtype=torch.uint8)
        for i, output in enumerate(score_outputs):
            try:
                score = int(json.loads(output.outputs[0].text)["score"])
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
                score = 0
                num_score_parse_failures += 1
            answer_scores[i] = score
            answer_table[i]["score"] = score
        score_duration = time.time() - score_start

        correct = answer_scores[answer_ids.long()]                    # [V, Q, K] uint8
        p_correct = correct.float().mean(dim=-1)                      # [V, Q]
        mean_score = correct.float().sum(dim=1).mean(dim=-1)          # [V] quiz score (out of Q), averaged over the K samples

        logger.info(f"[{meeting_name}] Base mean quiz score {mean_score[0]:.2f}/{num_q}; median over versions {mean_score[1:].median():.2f}; "
                    f"{num_truncated} truncated answers; {num_score_parse_failures} judge parse failures")

        torch.save({
            "correct": correct,              # [V, Q, K] 0/1 per sample; row 0 = base, row j+1 = line j clarified
            "p_correct": p_correct,          # [V, Q] fraction of the K samples graded correct
            "mean_score": mean_score,        # [V] average quiz score across the K samples
            "answer_ids": answer_ids,        # [V, Q, K] index into answer_table
            "answer_table": answer_table,    # distinct answers: {"q_idx", "answer", "score"}
            "is_noop": torch.tensor(is_noop),  # [N] True where GT line == ASR line (row j+1 copies the base)
            "quiz": quiz,
            "stats": {
                "num_truncated_answers": num_truncated,
                "num_score_parse_failures": num_score_parse_failures,
                "answer_duration_sec": answer_duration,
                "score_duration_sec": score_duration,
                "total_duration_sec": time.time() - meeting_start,
                "base_prompt_tokens": base_prompt_len,
                "prefix_cache_hit_rate": num_cached_tokens / max(num_prompt_tokens, 1),
            },
            "config": vars(args),
        }, save_path)

        logger.info(f"[{meeting_name}] Finished in {time.time() - meeting_start:.0f}s (answering {answer_duration:.0f}s, scoring {score_duration:.0f}s). Saved to {save_path}")

        del answer_texts, answer_table, answer_ids, score_messages, score_outputs
        gc.collect()

    logger.info("Destroying LLM...")
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("All tasks completed successfully!")
