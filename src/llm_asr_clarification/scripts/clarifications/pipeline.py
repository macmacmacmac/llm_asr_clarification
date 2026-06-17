import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import ipdb
import random
from llm_asr_clarification.models import OracleTranscript, OpenAIWrapper
from typing import Tuple, List
import re
from llm_asr_clarification.constants.clarification_prompts import (
    SUMMARIZER_SYS_PROMPT, SUMMARIZER_USER_PROMPT,
    CHOOSER_SYS_PROMPT, CHOOSER_USER_PROMPT
)

SAMPLING_RATE = 16_000
GROUND_TRUTH_TRANSCRIPT = "parsed_diarized_gt.txt"


# ┌───────────────────────────────────────────────┐
# │         CLARIFICATION CHOICE STRATEGY         │
# └───────────────────────────────────────────────┘
def extract_timestamps(line):
    # Extract all sequences of digits from the string
    numbers = re.findall(r'\d+', line)

    # Get the first two numbers and convert them to integers
    start_time = int(numbers[0])
    end_time = int(numbers[1])

    return start_time, end_time


def generate_summary(idx: int, transcript_lines: List[str]):
    # Get the transcript lines upto idx
    prefix_text = "".join(transcript_lines[:idx])

    # Prepare the user prompt
    user_prompt = SUMMARIZER_USER_PROMPT.format(
        input_meeting_transcription = prefix_text
    )

    # Generate summary
    generated_summary = SUMMARY_MODEL.prompt_chatgpt(
        prompt = user_prompt,
        max_tokens=1024
    )

    return generated_summary




# ┌───────────────────────────────────────────────┐
# │         CLARIFICATION CHOICE STRATEGY         │
# └───────────────────────────────────────────────┘
def choose_option(idx_pair: Tuple[int], transcript_lines: List[str]):
    match STRATEGY:
        case "RANDOM":
            return choose_randomly(idx_pair)
        case "LLM-ORIG-CTX":
            return choose_using_llm(idx_pair, transcript_lines)
        case "LLM-GT-CTX":
            pass
        case _:
            return choose_randomly(idx_pair)


def choose_randomly(idx_pair: Tuple):
    return random.choice(idx_pair)

def choose_using_llm(idx_pair: Tuple, transcript_lines: List[str]):
    idx0 = idx_pair[0]
    idx1 = idx_pair[1]

    # Generate Summaries
    summary0 = generate_summary(idx0, transcript_lines)
    summary1 = generate_summary(idx1, transcript_lines)
    
    # Fetch Transcriptions
    transcription0 = transcript_lines[idx0]
    transcription1 = transcript_lines[idx1]

    # Prepare prompt
    user_prompt = CHOOSER_USER_PROMPT.format(
        idx0 = idx0,
        idx1 = idx1,
        context0 = summary0,
        context1 = summary1,
        transcription0 = transcription0,
        transcription1 = transcription1
    )

    # Ask LLM
    choice_idx = CHOOSER_MODEL.prompt_chatgpt(
        prompt = user_prompt
    )

    # Convert to int
    try:
        choice_idx = int(choice_idx)
        if choice_idx not in idx_pair:
            logger.warning(f"LLM chose an idx not part of the idx_pair: {idx_pair}! Defaulting to idx_pair[0].")
            choice_idx = idx_pair[0]
    except Exception as err:
        logger.error(f"Error occurred while parsing choice_idx into a int: {err}")


    return choice_idx



# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--transcript-file", type=str, default="whisper_tiny_diarized_transcript.txt")
    parser.add_argument("--meeting-name", type=str, default="")
    parser.add_argument("--strategy", type=str, default="RANDOM")
    parser.add_argument("--seed", type=int, default=47)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRANSCRIPT_FILE = args.transcript_file
    global STRATEGY
    STRATEGY = args.strategy
    SEED = args.seed

    random.seed(SEED)


    # Global Variables
    global SUMMARY_MODEL
    global CHOOSER_MODEL
    SUMMARY_MODEL = OpenAIWrapper(
        system_prompt=SUMMARIZER_SYS_PROMPT
    )
    CHOOSER_MODEL = OpenAIWrapper(
        system_prompt=CHOOSER_SYS_PROMPT
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

            transcript_path = meeting_folder / "transcripts" / TRANSCRIPT_FILE
            oracle_transcript_path = meeting_folder / "transcripts" / GROUND_TRUTH_TRANSCRIPT

            # Fetch the Oracle Transcript
            oracle_transcript = OracleTranscript(
                transcript_path=oracle_transcript_path,
                logger=logger
            )
            with open(transcript_path, "r") as f:
                transcript_content = f.read()
            original_transcript_lines = transcript_content.split("\n")
            updated_transcript_lines = original_transcript_lines

            # Select 20 random lines
            random_line_idxs = random.choices(range(0, len(original_transcript_lines)), k = 20)
            
            # Group them into 10 Pairs
            random_idx_pairs = list(zip(random_line_idxs[::2], random_line_idxs[1::2]))

            # For each idx pair
            for idx_pair in random_idx_pairs:

                # Choose 1 from the pair
                chosen_idx = choose_option(idx_pair, original_transcript_lines)
                chosen_line = original_transcript_lines[chosen_idx]

                # Extract timestamps for the chosen line
                start_time, end_time = extract_timestamps(chosen_line)
                
                # Use Oracle Transcript to fetch timestamp oriented transcription
                oracle_lines = oracle_transcript.get_oracle_transcription(start_time=start_time, end_time=end_time)

                # Add extracted timestamps to the oracle lines
                oracle_lines = [f"({start_time} - {end_time}){line}" for line in oracle_lines]

                # Update transcript line with the combined oracle lines
                updated_transcript_lines[chosen_idx] = "".join(oracle_lines).strip()


                logger.info(f"Clarified timestamps: {start_time} - {end_time}")



            # ┌───────────────────────────────────────────────┐
            # │                     SAVE                      │
            # └───────────────────────────────────────────────┘
            clarified_file_name = TRANSCRIPT_FILE.split(".")[0] + f"_{STRATEGY.lower()}_clarify.txt"
            fixed_transcript_file_path = meeting_folder / "transcripts" / clarified_file_name
            with open(fixed_transcript_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(updated_transcript_lines))
                
            logger.info(f"Saved transcript for {fixed_transcript_file_path}\n\n")










            