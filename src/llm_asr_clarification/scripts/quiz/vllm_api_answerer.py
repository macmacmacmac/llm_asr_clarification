import os
import argparse
from dotenv import load_dotenv
from llm_asr_clarification import get_logger
from tqdm.auto import tqdm
import json
from filelock import FileLock
from collections import defaultdict
from pydantic import BaseModel
import asyncio
from openai import AsyncOpenAI

# ── Pydantic schema for structured output ──────────────────────────────────────
class AnswerResponse(BaseModel):
    answer: str

# ── Prompts ────────────────────────────────────────────────────────────────────
VLLM_ANSWER_SYSTEM_PROMPT = """You are an expert at paying attention to meetings and answering quiz questions about them.
You will be shown a transcription from a meeting and a single quiz question.
Your task is to answer the question based only on the information in the transcript.

Output Format:
Return ONLY a single JSON object with the key "answer". Do not include any preamble or extra text.
{
  "answer": "your answer here"
}"""

VLLM_ANSWER_USER_PROMPT = """## Transcript:
{transcript}

## Question:
{question}

Output JSON Answer:
"""

VLLM_BASELINE_ANSWER_SYSTEM_PROMPT = """You are an expert at answering quizzes meant to test your understanding on the meetings from the AMI Corpus Dataset.
You will be provided with the name of the relevant meeting from the AMI Corpus Dataset. Your task is to answer a single quiz question that only someone who closely understands that meeting will be able to answer.

Output Format:
Return ONLY a single JSON object with the key "answer". Do not include any preamble or extra text.
{
  "answer": "your answer here"
}"""

VLLM_BASELINE_ANSWER_USER_PROMPT = """## Meeting Name:
{meeting_name}

## Question:
{question}

Output JSON Answer:
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
            max_tokens=4096,
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "AnswerResponse", 
                    "schema": json_schema_dict,
                    "strict": True
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
    parser.add_argument("--baseline-prompting", action="store_true")
    parser.add_argument("--do-all-meetings", action="store_true")
    parser.add_argument("--meeting-name", type=str, default="ES2005d")


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

        transcript_path = os.path.join(meeting_path, "transcripts", f"{args.transcript_file}.txt")
        quiz_path = os.path.join(meeting_path, "quiz", "vllm_quiz.json")

        if not args.baseline_prompting:
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
            except Exception as e:
                logger.error(f"Could not read transcript for {meeting_name}: {e}")
                continue

        try:
            lock = FileLock(f"{quiz_path}.lock")
            with lock:
                with open(quiz_path, "r", encoding="utf-8") as f:
                    quiz = json.loads(f.read())
        except Exception as e:
            logger.error(f"Something failed, couldn't read quiz file for {meeting_name}: {e}")
            continue

        for q_idx, q in enumerate(quiz):
            if args.baseline_prompting:
                all_global_messages.append([
                    {"role": "system", "content": VLLM_BASELINE_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": VLLM_BASELINE_ANSWER_USER_PROMPT.format(
                        meeting_name=meeting_name,
                        question=q["question"],
                    )},
                ])
            else:
                all_global_messages.append([
                    {"role": "system", "content": VLLM_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": VLLM_ANSWER_USER_PROMPT.format(
                        transcript=transcript_text,
                        question=q["question"],
                    )},
                ])
            request_metadata.append({
                "meeting_name": meeting_name,
                "quiz_path": quiz_path,
                "q_idx": q_idx
            })
            
    if not all_global_messages:
        logger.warning("No valid questions found to answer. Exiting.")
        return

    # PASS 2: Batch Inference via local API
    logger.info(f"Sending {len(all_global_messages)} total questions to vLLM local API in a single batch...")
    try:
        global_outputs = asyncio.run(
            process_all_requests(all_global_messages, AnswerResponse.model_json_schema(), args.model_to_use)
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
                ans = "n/a"
            else:
                response_text = output.choices[0].message.content
                result = json.loads(response_text)
                ans = result.get("answer", "n/a")
        except Exception as e:
            logger.warning(
                f"Could not parse response for {meta['meeting_name']}.\n"
                f"Error: {e}"
            )
            ans = "n/a"
        outputs_by_quiz[meta["quiz_path"]].append((meta["q_idx"], ans))

    if args.baseline_prompting:
        answer_field = "answer_using_baseline_prompting"
    else:
        answer_field = f"answer_using_{args.transcript_file}"
        
    logger.info(f"Writing answers back to disk for {len(outputs_by_quiz)} meetings...")
    for quiz_path, results in tqdm(outputs_by_quiz.items(), desc="Writing results"):
        try:
            lock = FileLock(f"{quiz_path}.lock")
            with lock:
                with open(quiz_path, "r", encoding="utf-8") as f:
                    latest_quiz = json.loads(f.read())
                for q_idx, ans in results:
                    latest_quiz[q_idx][answer_field] = ans
                with open(quiz_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(latest_quiz, indent=4))
        except Exception as e:
            logger.error(f"Failed to write answers for {quiz_path}: {e}")

    logger.info("Answer generation complete.")
