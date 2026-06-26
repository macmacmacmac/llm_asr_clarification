import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import ipdb
from llm_asr_clarification.models import OpenAIWrapper
from llm_asr_clarification.constants import SAMPLE_MEETINGS
from llm_asr_clarification.constants.clarification_prompts import SUMMARIZER_SYS_PROMPT, SUMMARIZER_USER_PROMPT_TEMPLATE
from llm_asr_clarification.constants.filtration_prompts import (
    MISTRANSCRIPTIONS_SYS_PROMPT, MISTRANSCRIPTIONS_USER_PROMPT,
    IMPORTANT_SYS_PROMPT, IMPORTANT_USER_PROMPT,
    UNSMOOTHABLE_SYS_PROMPT, UNSMOOTHABLE_USER_PROMPT
)
from typing import List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

GROUND_TRUTH_TRANSCRIPT = "parsed_diarized_gt.txt"


# ┌───────────────────────────────────────────────┐
# │                   HELPER METHODS              │
# └───────────────────────────────────────────────┘
def generate_summaries(passages: List[List[str]]) -> List[str]:
    summaries = []
    for i, passage in enumerate(passages):
        prev_passages = passages[:i]
        if len(prev_passages) > 0:
            # Combine all prev passages into text
            prev_text = ""
            for passage in prev_passages:
                prev_text = prev_text + "\n" + "\n".join(passage)

            # Prepare the user prompt
            user_prompt = SUMMARIZER_USER_PROMPT_TEMPLATE.format(
                input_meeting_transcription = prev_text
            )

            # Generate summary
            response = SUMMARY_MODEL.chat_completion(
                messages = [
                    {
                        "role": "system",
                        "content": SUMMARIZER_SYS_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                    
                ],
                max_tokens=2048,
                temperature=0.5
            )
            summary = response.choices[0].message.content
            summaries.append(summary)

        else:
            summaries.append("No Context is available, this is the beginning passage.")

    return summaries

        
    


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--transcript-file", type=str, default="whisper_tiny_diarized_transcript.txt")
    parser.add_argument("--do-sample-meetings", action="store_true")
    parser.add_argument("--meeting-name", type=str, default="")
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRANSCRIPT_FILE = args.transcript_file


    # Global Variables
    global MISTRANSCRIPTIONS_MODEL
    MISTRANSCRIPTIONS_MODEL = OpenAIWrapper(
        system_prompt=""
    )

    global IMPORTANT_MODEL
    IMPORTANT_MODEL = OpenAIWrapper(
        system_prompt=""
    )

    global UNSMOOTHABLE_MODEL
    UNSMOOTHABLE_MODEL = OpenAIWrapper(
        system_prompt=""
    )

    global SUMMARY_MODEL
    SUMMARY_MODEL = InferenceClient(
        model = "Qwen/Qwen2.5-7B-Instruct",
        token = os.getenv("HF_TOKEN")
    )
    
    
    # Init Logger
    global logger
    logger = get_logger(exp_name)    
    logger.info(f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}")

    # Log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(
        f"Received the following arguments:\n{received_args_log}"
    )


    # ┌───────────────────────────────────────────────┐
    # │                   LOAD DATA                   │
    # └───────────────────────────────────────────────┘
    if args.meeting_name:
        meeting_folders=[DATASET_PATH / args.meeting_name]
    elif args.do_sample_meetings:
        meeting_folders = [f for f in DATASET_PATH.iterdir()
                          if (f.is_dir() and
                              f.name in SAMPLE_MEETINGS)]
    else:
        # Fetch all dataset meeting folders
        meeting_folders = [f for f in DATASET_PATH.iterdir() 
                            if (f.is_dir() and 
                                f.name not in ["ami_public_manual_1.6.2", "xinlu_data"])]
        
        

    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):
            logger.info(f"Processing meeting {meeting_folder.name}")

            # Read the Transcript
            transcript_path = meeting_folder / "transcripts" / TRANSCRIPT_FILE
            with open(transcript_path, "r") as f:
                transcript_lines = f.read().split("\n")
            logger.info(f"Read {len(transcript_lines)} lines")

            # Break Transcript into passages of 10 lines
            passages = [transcript_lines[i: i + 10] for i in range(0, len(transcript_lines), 10)]
            logger.info(f"Found {len(passages)} for further processing")

            # Generate Summaries for all passages
            summaries = generate_summaries(passages)
            ipdb.set_trace()
            logger.info(f"Generated summary for {len(summaries)} passages")
            

            
            




            
            