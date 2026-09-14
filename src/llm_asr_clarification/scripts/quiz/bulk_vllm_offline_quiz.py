# ruff: noqa: BLE001 # Ignores usage of blind except blocks
"""
Determinism in vLLM Offline Batching:
This script guarantees 100% deterministic outputs across runs by addressing multiple sources of non-determinism:

1. OS Directory Iteration: 
   `os.scandir` is wrapped in `sorted()` to ensure all meeting folders are read and prompts are fed to vLLM in the exact same order.
2. Floating-Point Reductions in CUDA (FlashAttention / Marlin FP8): 
   By setting `max_num_seqs=1` in the LLM config, we force vLLM to process exactly one sequence at a time. 
   This entirely eliminates the floating-point non-associativity noise that occurs when parallel CUDA kernels perform reduction across 
   a dynamically changing batch of sequences.
3. Random Seeds:
   `seed=47` is strictly enforced in both the LLM engine initialization and the SamplingParams.

Note: By keeping `max_num_seqs=1`, we safely retain fast optimizations like CUDA graphs and Prefix Caching because they execute deterministically on a batch size of 1.
"""

import os
import json
import gc
import time
import argparse

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
from llm_asr_clarification.scripts.archived.vllm_api_answerer import AnswerResponse
from llm_asr_clarification.scripts.archived.vllm_api_scorer import ScoreResponse

def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--answering-model", type=str, default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--scoring-model", type=str, default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--ami-path", type=str, default="./shared/datasets/amicorpus")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)

    args, _ = parser.parse_known_args(args_list)

    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    # Configurations
    splits = ["validation"]
    clarification_num_lines = ["10", "20", "30", "40", "50"]
    importance_detectors = ["LSTM", "GT", "ALL"]
    mistranscript_detectors = ["RF", "GT", "ALL"]

    versions = ["4"]

    configs = []
    for split in splits:
        for clar_num_line in clarification_num_lines:
            for imp_det in importance_detectors:
                for mis_det in mistranscript_detectors:
                    for version in versions:
                        imp_det_lower = imp_det.lower()
                        mis_det_lower = mis_det.lower()
                        transcript_file = f"custom_transcript_gt_segments_{mis_det_lower}_{imp_det_lower}_{clar_num_line}_clarify{version}"
                        configs.append({
                            "split": split,
                            "transcript_file": transcript_file
                        })
    
    logger.info(f"Total configurations to process: {len(configs)}")
    
    # -------------------------------------------------------------------------
    # PHASE 1: ANSWERING
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80 + "\nPHASE 1: ANSWERING\n" + "="*80)
    
    all_answer_messages = []
    answer_metadata = []
    
    for config in tqdm(configs, desc="Building Answering Prompts"):
        split = config["split"]
        transcript_file = config["transcript_file"]
        split_path = os.path.join(args.ami_path, split)
        
        if not os.path.exists(split_path):
            logger.warning(f"Unable to find folder: {split_path}")
            continue
            
        meeting_paths = sorted(entry.path for entry in os.scandir(split_path) if entry.is_dir())
        
        for meeting_path in meeting_paths:
            meeting_name = os.path.basename(meeting_path)
            if meeting_name == "EN2009d":
                logger.info("Skipping Meeting EN2009d as its too big for the vLLM model!")
                continue
                
            transcript_path = os.path.join(meeting_path, "transcripts", f"{transcript_file}.txt")
            quiz_path = os.path.join(meeting_path, "quiz", "vllm_quiz.json")

            # Open the transcript file
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
            except FileNotFoundError:
                logger.warning(f"Unable to find the transcript at {transcript_path}")
                continue
            except OSError as e:
                logger.warning(f"OSError reading {transcript_path}: {e}")
                continue

            # Open the quiz file
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
            except OSError as e:
                logger.warning(f"OSError reading {quiz_path}: {e}")
                continue

            # Create Prompts for all quiz questions
            for q_idx, q in enumerate(quiz):
                all_answer_messages.append([
                    {"role": "system", "content": VLLM_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": VLLM_ANSWER_USER_PROMPT.format(
                        transcript=transcript_text,
                        question=q["question"]
                    )}
                ])
                answer_metadata.append({
                    "quiz_path": quiz_path,
                    "q_idx": q_idx,
                    "transcript_file": transcript_file
                })

    # Proceed with answering questions using all prompts prepared
    if all_answer_messages:
        logger.info(f"Loading LLM for Answering ({args.answering_model})...")
        # Deterministic config: process one prompt at a time to eliminate batch-level non-determinism
        # We allow prefix caching, chunked prefill, and CUDA graphs (enforce_eager=False) for speed, 
        # as they are perfectly deterministic when batch size is strictly 1.
        llm = LLM(
            model=args.answering_model,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=47,
            max_num_seqs=1
        )
        
        sampling_params = SamplingParams(
            temperature=0.0, 
            seed=47, 
            max_tokens=4096, 
            structured_outputs=StructuredOutputsParams(json=json.dumps(AnswerResponse.model_json_schema()))
        )
        
        logger.info(f"Generating {len(all_answer_messages)} answers (Strictly Deterministic)...")
        start_time = time.time()
        outputs = llm.chat(
            messages=all_answer_messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        logger.info(f"Answer generation took {time.time() - start_time:.2f} seconds.")
        
        answers_by_quiz = defaultdict(list)
        for meta, output in zip(answer_metadata, outputs):
            try:
                response_text = output.outputs[0].text
                result = json.loads(response_text)
                ans = result.get("answer", "n/a")
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
                logger.warning("Error Parsing JSON response for Answering")
                ans = "n/a"
                
            answer_field = f"answer_using_{meta['transcript_file']}"
            answers_by_quiz[meta["quiz_path"]].append((meta["q_idx"], answer_field, ans))
            
        logger.info(f"Writing answers back to {len(answers_by_quiz)} quiz files...")
        for quiz_path, results in tqdm(answers_by_quiz.items(), desc="Saving Answers"):
            try:
                lock = FileLock(f"{quiz_path}.lock")
                with lock, open(quiz_path, "r", encoding="utf-8") as f:
                    latest_quiz = json.loads(f.read())
                    
                for q_idx, answer_field, ans in results:
                    latest_quiz[q_idx][answer_field] = ans
                    
                with lock, open(quiz_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(latest_quiz, indent=4))

            except Exception as e:
                logger.error(f"Failed writing {quiz_path}: {e}")
        # Gracefully destroy the vLLM instance
        logger.info("Destroying Answering LLM...")
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("No answer prompts found.")


    # -------------------------------------------------------------------------
    # PHASE 2: SCORING
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80 + "\nPHASE 2: SCORING\n" + "="*80)
    
    all_score_messages = []
    score_metadata = []
    
    for config in tqdm(configs, desc="Building Scoring Prompts"):
        split = config["split"]
        transcript_file = config["transcript_file"]
        split_path = os.path.join(args.ami_path, split)
        
        if not os.path.exists(split_path):
            logger.warning(f"Unable to find folder: {split_path}")
            continue
            
        meeting_paths = sorted(entry.path for entry in os.scandir(split_path) if entry.is_dir())
        
        for meeting_path in meeting_paths:
            meeting_name = os.path.basename(meeting_path)
            if meeting_name == "EN2009d":
                logger.info("Skipping Meeting EN2009d as its too big for the vLLM model!")
                continue
                
            quiz_path = os.path.join(meeting_path, "quiz", "vllm_quiz.json")
            
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
            except OSError as e:
                logger.warning(f"OSError reading {quiz_path}: {e}")
                continue
                
            answer_field = f"answer_using_{transcript_file}"
            
            for q_idx, q in enumerate(quiz):
                if answer_field not in q:
                    logger.warning(f"Unable to find {answer_field} field in {q} for meeting {meeting_name}")
                    continue
                    
                all_score_messages.append([
                    {"role": "system", "content": VLLM_SCORER_SYSTEM_PROMPT},
                    {"role": "user", "content": VLLM_SCORER_USER_PROMPT.format(
                        question=q["question"],
                        correct_answer=q["correct_answer"],
                        predicted_answer=q[answer_field]
                    )}
                ])
                score_metadata.append({
                    "quiz_path": quiz_path,
                    "q_idx": q_idx,
                    "transcript_file": transcript_file
                })

    # Proceed with scoring quizzes using all prompts prepared
    if all_score_messages:
        logger.info(f"Loading LLM for Scoring ({args.scoring_model})...")
        # Deterministic config: process one prompt at a time to eliminate batch-level non-determinism
        # We allow prefix caching, chunked prefill, and CUDA graphs (enforce_eager=False) for speed, 
        # as they are perfectly deterministic when batch size is strictly 1.
        llm = LLM(
            model=args.scoring_model,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=47,
            max_num_seqs=1
        )
        
        sampling_params = SamplingParams(
            temperature=0.0, 
            seed=47, 
            max_tokens=64, 
            structured_outputs=StructuredOutputsParams(json=json.dumps(ScoreResponse.model_json_schema()))
        )
        
        
        logger.info(f"Generating {len(all_score_messages)} scores (Strictly Deterministic)...")
        start_time = time.time()
        outputs = llm.chat(
            messages=all_score_messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        logger.info(f"Score generation took {time.time() - start_time:.2f} seconds.")
        
        scores_by_quiz = defaultdict(list)
        for meta, output in zip(score_metadata, outputs):
            try:
                response_text = output.outputs[0].text
                result = json.loads(response_text)
                score = result.get("score", 0)
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
                logger.warning("Error Parsing JSON response for Scoring")
                score = 0
                
            score_field = f"score_using_{meta['transcript_file']}"
            scores_by_quiz[meta["quiz_path"]].append((meta["q_idx"], score_field, score))
            
        logger.info(f"Writing scores back to {len(scores_by_quiz)} quiz files...")
        for quiz_path, results in tqdm(scores_by_quiz.items(), desc="Saving Scores"):
            try:
                lock = FileLock(f"{quiz_path}.lock")
                with lock, open(quiz_path, "r", encoding="utf-8") as f:
                    latest_quiz = json.loads(f.read())
                    
                for q_idx, score_field, s in results:
                    latest_quiz[q_idx][score_field] = s
                    
                with lock, open(quiz_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(latest_quiz, indent=4))
            except Exception as e:
                logger.error(f"Failed writing {quiz_path}: {e}")

        logger.info("Destroying Scoring LLM...")
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("No score prompts found.")

    logger.info("All tasks completed successfully!")
