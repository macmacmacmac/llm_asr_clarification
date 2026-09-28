# ruff: noqa: BLE001 # Ignores usage of blind except blocks
"""
    This is to be a script for generating gold labels for importance detection.
    The script works as follow
    1. Within a given folder (pass as argument), it retrieves the list of meeting
        Project directory

        ./shared/datasets/amicorpus
            <split> (this is either 'train' or 'split')
                |--<meeting_name> (e.g 'EN2001a', etc)
                    |--transcripts
                      |--<ASR transcript>.txt
                      |--<ground truth transcript>.txt
                |--<meeting_name (e.g. 'EN2001b', etc)>
                    ...
                ...

        The script should be passed as an argument and takes either 'train' or 'validation'

        The meeting ASR transcript is always named 'custom_transcript_gt_segments.txt'
        The groundtruth transcript is always named 'parsed_diarized_gt.txt'

    2. For each meeting transcript, we will iteratively, starting from the first segment to the last segment in the meeting replace it with corresponding segments from the ground truth transcript.
    3. Segment replacement is done by a window size and a stride argument. E.g. a window = 1, stride = 1 means replacing one segment at a time, etc. At every iteration, we will save this version of the transcript in a list of strings.
    4. Once we reach the end of the transcript, we will compute the quiz answers and quiz score using vllm for all the versions of the transcripts at once. After this step, we should now have another parallel list of quiz scores for each version of the transcript. Do NOT save to the quiz json. This is NOT necessary.
    5. If the i-th iteration of a transcript has a higher score than the i-1th iteration, then that means the segments that we replaced in the i-th version were important. In a dictionary of {"<meeting_name>": [<boolean index of important indices]} save which segment indices were important. This dictionary should be saved to "./shared/datasets/gold_labels/<split>.pt" and saving should be done after each meeting is finished.
    6. Repeat for all meetings in the split.
    
"""

import os
import json
import gc
import time
import argparse
import ipdb

import torch
from tqdm.auto import tqdm
from filelock import FileLock
from collections import defaultdict

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.sampling_params import StructuredOutputsParams

from llm_asr_clarification import get_logger
from llm_asr_clarification.constants.quiz_prompts import (
    VLLM_ANSWER_SYSTEM_PROMPT, VLLM_ANSWER_USER_PROMPT,
    VLLM_SCORER_SYSTEM_PROMPT, VLLM_SCORER_USER_PROMPT
)
from pydantic import BaseModel
from typing import Literal


# Pydantic Schemas for structured outputs
class AnswerResponse(BaseModel):
    answer: str

class ScoreResponse(BaseModel):
    score: Literal[0, 1]



def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-FP8")
    parser.add_argument("--ami-path", type=str, default="./shared/datasets/amicorpus")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--which_half", type=str, default="full")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=40960)

    args, _ = parser.parse_known_args(args_list)

    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    logger.info(f"Loading LLM ({args.model})...")
    # We allow prefix caching, chunked prefill, and CUDA graphs (enforce_eager=False) for speed, 
    # as they are perfectly deterministic when batch size is strictly 1.
    #llm = LLM(
    #    model=args.model,
    #    max_model_len=args.max_model_len,
    #    tensor_parallel_size=args.tensor_parallel_size,
    #    seed=47,
    #    # gpu_memory_utilization=0.95
    #    #max_num_seqs=1
    #)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=47,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
    )
    
    answer_sampling_params = SamplingParams(
        temperature=0.0, 
        seed=47, 
        max_tokens=512, 
        structured_outputs=StructuredOutputsParams(json=json.dumps(AnswerResponse.model_json_schema()))
    )
    
    score_sampling_params = SamplingParams(
        temperature=0.0, 
        seed=47, 
        max_tokens=64, 
        structured_outputs=StructuredOutputsParams(json=json.dumps(ScoreResponse.model_json_schema()))
    )

    gold_labels = {}
    save_path = f"./shared/datasets/gold_labels/{args.split}_{args.model.split("/")[-1]}_{args.window}w_{args.stride}s_{args.which_half}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        gold_labels = torch.load(save_path)
        logger.info(f"Loaded existing gold labels for {len(gold_labels)} meetings.")

    split_path = os.path.join(args.ami_path, args.split)
    if not os.path.exists(split_path):
        logger.error(f"Unable to find folder: {split_path}")
        return

    meeting_paths = sorted(entry.path for entry in os.scandir(split_path) if entry.is_dir())

    # Filtering meetings 
    meetings_too_long = [
        "EN2009d",
        "EN2001a",
        "EN2005a",
        "IN1013",
        "IN1016",
        "EN2006a",
        "TS3006d"
    ]
    # meeting_paths = [meeting_path for meeting_path in meeting_paths if 'TS3005d' in meeting_path]

    meeting_paths = [meeting_path for meeting_path in meeting_paths if meeting_path not in meetings_too_long]
    if args.which_half == "first":
        meeting_paths = meeting_paths[:len(meeting_paths)//2]
    elif args.which_half == "second":
        meeting_paths = meeting_paths[len(meeting_paths)//2:]
    # ipdb.set_trace()

    for meeting_path in tqdm(meeting_paths, desc="Processing Meetings"):
        meeting_name = os.path.basename(meeting_path)
        if meeting_name in gold_labels:
            logger.info(f"Skipping {meeting_name}, already processed.")
            continue
            
        if meeting_name == "EN2009d":
            logger.info("Skipping Meeting EN2009d as its too big for the vLLM model!")
            continue

        asr_path = os.path.join(meeting_path, "transcripts", "custom_transcript_gt_segments.txt")
        gt_path = os.path.join(meeting_path, "transcripts", "parsed_diarized_gt.txt")
        quiz_path = os.path.join(meeting_path, "quiz", "vllm_quiz.json")

        try:
            with open(asr_path, "r", encoding="utf-8") as f:
                asr_lines = [line.strip() for line in f.readlines()]
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError as e:
            logger.warning(f"Unable to find transcripts for {meeting_name}: {e}")
            continue

        if len(asr_lines) != len(gt_lines):
            logger.warning(f"Line count mismatch for {meeting_name}: ASR {len(asr_lines)} != GT {len(gt_lines)}")
            continue

        try:
            lock = FileLock(f"{quiz_path}.lock")
            with lock, open(quiz_path, "r", encoding="utf-8") as f:
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

        # # 2. Build Transcript Iterations (Cumulative)
        # transcript_versions = ["\n".join(asr_lines)]
        # current_lines = list(asr_lines)
        # replaced_indices_per_iter = [[]]
        
        # for i in range(0, len(asr_lines), args.stride):
        #     replaced = []
        #     for j in range(i, min(i + args.window, len(asr_lines))):
        #         current_lines[j] = gt_lines[j]
        #         replaced.append(j)
        #     transcript_versions.append("\n".join(current_lines))
        #     replaced_indices_per_iter.append(replaced)

        # 2. Build Transcript Iterations (Non-Cumulative)
        base_transcript = "\n".join(asr_lines)
        transcript_versions = [base_transcript]
        replaced_indices_per_iter = [[]]

        for i in range(0, len(asr_lines), args.stride):
            gt_window = gt_lines[i:i+args.stride]

            asr_copy = asr_lines.copy()
            asr_copy[i:i+args.stride] = gt_window
            transcript_versions.append("\n".join(asr_copy))

            replaced_idxs = list(range(i, i+args.stride))
            replaced_indices_per_iter.append(replaced_idxs)

            # ipdb.set_trace()

        # 3. Answering Phase
        logger.info(f"[{meeting_name}] Building answer prompts for {len(transcript_versions)} iterations...")
        all_answer_messages = []
        for v_idx, transcript_text in enumerate(transcript_versions):
            for q_idx, q in enumerate(quiz):
                all_answer_messages.append([
                    {"role": "system", "content": VLLM_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": VLLM_ANSWER_USER_PROMPT.format(
                        transcript=transcript_text,
                        question=q["question"]
                    )}
                ])
                
        logger.info(f"[{meeting_name}] Generating {len(all_answer_messages)} answers...")
        start_time = time.time()
        answer_outputs = llm.chat(
            messages=all_answer_messages,
            sampling_params=answer_sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        logger.info(f"[{meeting_name}] Answer generation took {time.time() - start_time:.2f} seconds.")
        
        # 4. Scoring Phase
        all_score_messages = []
        for i, output in enumerate(answer_outputs):
            v_idx = i // len(quiz)
            q_idx = i % len(quiz)
            q = quiz[q_idx]
            
            try:
                response_text = output.outputs[0].text
                result = json.loads(response_text)
                ans = result.get("answer", "n/a")
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
                ans = "n/a"
                
            all_score_messages.append([
                {"role": "system", "content": VLLM_SCORER_SYSTEM_PROMPT},
                {"role": "user", "content": VLLM_SCORER_USER_PROMPT.format(
                    question=q["question"],
                    correct_answer=q["correct_answer"],
                    predicted_answer=ans
                )}
            ])

        logger.info(f"[{meeting_name}] Generating {len(all_score_messages)} scores...")
        start_time = time.time()
        score_outputs = llm.chat(
            messages=all_score_messages,
            sampling_params=score_sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        logger.info(f"[{meeting_name}] Score generation took {time.time() - start_time:.2f} seconds.")
        
        # Parse scores and aggregate per version
        version_scores = [0] * len(transcript_versions)
        for i, output in enumerate(score_outputs):
            v_idx = i // len(quiz)
            try:
                response_text = output.outputs[0].text
                result = json.loads(response_text)
                score = result.get("score", 0)
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
                score = 0
            version_scores[v_idx] += score
            
        # 5. Compute Importance
        is_important = [False] * len(asr_lines)
        for k in range(1, len(transcript_versions)):
            # if version_scores[k] > version_scores[k-1]: # For cumulative transcript versions
            if version_scores[k] > version_scores[0]: # For non-cumulative transcript versions
                for idx in replaced_indices_per_iter[k]:
                    is_important[idx] = True

        # 6. Save
        gold_labels[meeting_name] = is_important
        torch.save(gold_labels, save_path)

        # ipdb.set_trace()
        logger.info(f"[{meeting_name}] Finished! Saved importance mask with {sum(is_important)} true labels.")

    logger.info("Destroying LLM...")
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("All tasks completed successfully!")
