import os
import argparse
from dotenv import load_dotenv
from llm_asr_clarification import get_logger
from tqdm.auto import tqdm
import json
from filelock import FileLock
from collections import defaultdict
from pydantic import BaseModel
from typing import Literal
import asyncio
from openai import AsyncOpenAI

# ── Pydantic schema for structured output ──────────────────────────────────────
class ScoreResponse(BaseModel):
    score: Literal[0, 1]

# ── Prompts ────────────────────────────────────────────────────────────────────
VLLM_SCORER_SYSTEM_PROMPT = """You are an expert quiz grader. 
Your task is to score a single predicted answer against a correct answer by evaluating if it conveys the same core meaning, ignoring exact wording.

Scoring Rules:
- Award 1 (Correct): The predicted answer paraphrases, captures the essential meaning, or contains the core idea (even with extraneous info or different granularity). Focus on meaning.
- Award 0 (Incorrect): The predicted answer states a fundamentally different fact, contradicts the correct answer, is too vague, or says "I don't know".

Output Format:
Return ONLY a single JSON object. Do not include any preamble or extra text. Use the exact format below:
{
  "score": 0 | 1
}"""

VLLM_SCORER_USER_PROMPT = """Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}

Output JSON Score:
"""

async def process_all_requests(messages_list, json_schema_dict, model_name):
    client = AsyncOpenAI(
        api_key="EMPTY", 
        base_url="http://localhost:8000/v1"
    )
    
    async def fetch(msgs):
        return await client.chat.completions.create(
            model=model_name,
            messages=msgs,
            temperature=0.0,
            seed=47,
            max_tokens=64,
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "ScoreResponse", 
                    "schema": json_schema_dict
                }
            }
        )

    tasks = [fetch(msgs) for msgs in messages_list]
    return await asyncio.gather(*tasks, return_exceptions=True)


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-to-use", type=str, default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--ami-path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--transcript-file", type=str, default="parsed_diarized_gt")
    parser.add_argument("--do-all-meetings", action="store_true")
    parser.add_argument("--meeting-name", type=str, default="ES2005d")
    parser.add_argument("--baseline-prompting", action="store_true")
    # Kept for compatibility
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    args, _ = parser.parse_known_args(args_list)

    logger = get_logger(exp_name)
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(f"Received the following arguments:\n{received_args_log}")

    # Determine directories of meetings
    if args.do_all_meetings:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path)]
    else:
        meeting_paths = [f"{args.ami_path}/{args.meeting_name}"]

    # PASS 1: Accumulate all requests
    all_global_messages = []
    request_metadata = []

    for meeting_path in tqdm(meeting_paths, desc="Preparing prompts"):
        meeting_name = meeting_path.split("/")[-1]

        if meeting_name == "EN2009d":
            logger.info(f"Skipping {meeting_name} due to excessive context length (>40960).")
            continue

        quiz_path = os.path.join(meeting_path, "quiz", "vllm_quiz.json")

        try:
            lock = FileLock(f"{quiz_path}.lock")
            with lock:
                with open(quiz_path, "r", encoding="utf-8") as f:
                    quiz = json.loads(f.read())
        except Exception as e:
            logger.error(f"Something failed, couldn't read quiz file: {e}")
            continue

        if args.baseline_prompting:
            answer_field = "answer_using_baseline_prompting"
        else:
            answer_field = f"answer_using_{args.transcript_file}"

        for q_idx, q in enumerate(quiz):
            if answer_field not in q:
                continue
            all_global_messages.append([
                {"role": "system", "content": VLLM_SCORER_SYSTEM_PROMPT},
                {"role": "user", "content": VLLM_SCORER_USER_PROMPT.format(
                    question=q["question"],
                    correct_answer=q["correct_answer"],
                    predicted_answer=q[answer_field],
                )},
            ])
            request_metadata.append({
                "meeting_name": meeting_name,
                "quiz_path": quiz_path,
                "q_idx": q_idx
            })

    if not all_global_messages:
        logger.warning("No valid answers found to score. Exiting.")
        return

    # PASS 2: Batch Inference via local API
    logger.info(f"Sending {len(all_global_messages)} total questions to vLLM local API in a single batch...")
    try:
        global_outputs = asyncio.run(
            process_all_requests(all_global_messages, ScoreResponse.model_json_schema(), args.model_to_use)
        )
    except Exception as e:
        logger.error(f"vLLM API global inference failed: {e}")
        return

    # PASS 3: Write back to disk
    outputs_by_quiz = defaultdict(list)

    for meta, output in zip(request_metadata, global_outputs):
        try:
            if isinstance(output, Exception):
                logger.error(f"API Error for {meta['meeting_name']}: {output}")
                score = 0
            else:
                response_text = output.choices[0].message.content
                result = json.loads(response_text)
                score = result.get("score", 0)
        except Exception as e:
            logger.error(
                f"Could not parse response as JSON for {meta['meeting_name']}. Defaulting to 0.\n"
                f"Error: {e}"
            )
            score = 0
        outputs_by_quiz[meta["quiz_path"]].append((meta["q_idx"], score))

    if args.baseline_prompting:
        score_field = "score_using_baseline_prompting"
    else:
        score_field = f"score_using_{args.transcript_file}"

    logger.info(f"Writing scores back to disk for {len(outputs_by_quiz)} meetings...")
    for quiz_path, results in tqdm(outputs_by_quiz.items(), desc="Writing results"):
        try:
            lock = FileLock(f"{quiz_path}.lock")
            with lock:
                with open(quiz_path, "r", encoding="utf-8") as f:
                    latest_quiz = json.loads(f.read())
                for q_idx, s in results:
                    latest_quiz[q_idx][score_field] = s
                with open(quiz_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(latest_quiz, indent=4))
        except Exception as e:
            logger.error(f"Failed to write scores for {quiz_path}: {e}")

    logger.info("Scoring complete.")
