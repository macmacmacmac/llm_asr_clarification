import os
import argparse
from dotenv import load_dotenv
from llm_asr_clarification import get_logger
from tqdm.auto import tqdm
import json
from filelock import FileLock
from collections import defaultdict
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import torch


if torch.cuda.is_available():
    print(f"Total GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA GPU available.")

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


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)

    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-to-use", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--ami-path", type=str, default="./shared/datasets/amicorpus/train")
    parser.add_argument("--transcript-file", type=str, default="parsed_diarized_gt")
    parser.add_argument("--question-file", type=str, default="parsed_diarized_gt")
    parser.add_argument("--do-all-meetings", action="store_true")
    parser.add_argument("--meeting-name", type=str, default="ES2005d")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    args, _ = parser.parse_known_args(args_list)

    # Build the logger here
    logger = get_logger(exp_name)
    logger.info(
        f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}"
    )

    # log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(
        f"Received the following arguments:\n{received_args_log}"
    )

    #==============================================================================================

    # Load vLLM model once
    logger.info(f"Loading vLLM model: {args.model_to_use}")
    
    # Load HF_TOKEN from .env to authenticate with HuggingFace Hub
    load_dotenv()
    
    llm = LLM(
        model=args.model_to_use,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    logger.info("Model loaded successfully")

    # Configure sampling with guided JSON
    # Prepare guided decoding params
    json_schema = json.dumps(AnswerResponse.model_json_schema())
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
        structured_outputs=StructuredOutputsParams(json=json_schema)
    )

    # Determine directories of meetings
    if args.do_all_meetings:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path)]
    else:
        meeting_paths = [f"{args.ami_path}/{args.meeting_name}"]

    # PASS 1: Accumulate all requests
    all_global_messages = []
    request_metadata = []

    for meeting_path in tqdm(meeting_paths, desc="Preparing prompts"):
        # Extract Meeting Name
        meeting_name = meeting_path.split("/")[-1]

        # SKIP EN2009d (Outlier Context Length)
        # Reason: Qwen3-32B-FP8 has a native max_position_embeddings of 40960. EN2009d has ~41,000 tokens.
        # While vLLM can be forced to override this limit in Python, the underlying Triton/C++ GPU kernels 
        # hardcode the 40960 boundary and will throw an assertion crash (index out of bounds).
        # We skip it to ensure the pipeline succeeds for the other 135 meetings.
        if meeting_name == "EN2009d":
            logger.info(f"Skipping {meeting_name} due to excessive context length (>40960).")
            continue

        # Prep paths for transcript and quiz
        transcript_path = os.path.join(meeting_path, "transcripts", f"{args.transcript_file}.txt")
        quiz_path = os.path.join(meeting_path, "quiz", "vllm_quiz.json")

        # Read transcript
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        except Exception as e:
            logger.error(f"Could not read transcript for {meeting_name}: {e}")
            continue

        # Read quiz safely with a lock
        try:
            lock = FileLock(f"{quiz_path}.lock")
            with lock:
                with open(quiz_path, "r", encoding="utf-8") as f:
                    quiz = json.loads(f.read())
        except Exception as e:
            logger.error(f"Something failed, couldn't read quiz file for {meeting_name}: {e}")
            continue

        # Build all message lists for this meeting (one per question)
        for q_idx, q in enumerate(quiz):
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

    # PASS 2: Batch Inference
    logger.info(f"Sending {len(all_global_messages)} total questions to vLLM in a single batch...")
    try:
        global_outputs = llm.chat(
            messages=all_global_messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as e:
        logger.error(f"vLLM global inference failed: {e}")
        return

    # PASS 3: Write back to disk
    outputs_by_quiz = defaultdict(list)
    
    # Group results by meeting
    for meta, output in zip(request_metadata, global_outputs):
        try:
            result = json.loads(output.outputs[0].text)
            ans = result["answer"]
        except Exception:
            logger.warning(
                f"Could not parse response for {meta['meeting_name']}.\n"
                f"Response: {output.outputs[0].text}"
            )
            ans = "n/a"
        outputs_by_quiz[meta["quiz_path"]].append((meta["q_idx"], ans))

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

    # Explicit GPU cleanup so scorer can chain after this
    logger.info("Cleaning up GPU memory...")
    del llm
    import torch
    torch.cuda.empty_cache()
    logger.info("GPU memory freed")
