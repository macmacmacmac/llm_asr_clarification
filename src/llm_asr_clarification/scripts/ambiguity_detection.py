import os
import argparse
from llm_asr_clarification import get_logger, OpenAIWrapper
from llm_asr_clarification.models.prompts import AMBIGUITY_SYSTEM_PROMPT, AMBIGUITY_USER_PROMPT
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
import re
import ast
import ipdb
import json
# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing=================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_use", type=str, default="gpt-4o-mini")
    parser.add_argument("--ami_path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--transcript_file", type=str, default="whisper_tiny_diarized_transcript")
    parser.add_argument("--do_all_meetings", action="store_true")
    parser.set_defaults(do_all_meetings=False)
    parser.add_argument("--meeting_to_do", type=str, default="./datasets/amicorpus/ES2005d")
    
    args, _ = parser.parse_known_args(args_list)

    # Build the logger here
    # first arg is
    logger = get_logger(exp_name)    
    logger.info(
        f"{"="*100}\n\t\t\t\tRunning script: {exp_name}\n{"="*100}"
    )

    # log received args
    received_args_log = ""
    for arg, value in vars(args).items():
        received_args_log += f"|---> {arg}: {value}\n"
    logger.info(
        f"Received the following arguments:\n{received_args_log}"
    )

    #==============================================================================================
    # directories of meetings
    if args.do_all_meetings:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path) if entry.name not in ['ami_public_manual_1.6.2', 'xinlu_data']]
    else:
        meeting_paths = [args.meeting_to_do]

    for meeting_path in tqdm(meeting_paths):
        file_todo_path = os.path.join(meeting_path, "transcripts", f"{args.transcript_file}.txt")
        output_preds_path = os.path.join(meeting_path, "transcripts", f"ambiguity_preds_{args.transcript_file}.md")
        
        logger.info(f"I am doing this file: {file_todo_path}")
        
        chatgpt = OpenAIWrapper(logger=logger, system_prompt=AMBIGUITY_SYSTEM_PROMPT)

        # Read transcript
        with open(file_todo_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        markdown_chunks = []

        prev_lines = ["<Meeting start>"]

        for line in tqdm(transcript_text.split("\n")):

            prompt = AMBIGUITY_USER_PROMPT.format(
                transcript_context="\n".join(prev_lines),
                transcript_excerpt=line
            )

            response_text = chatgpt.prompt_chatgpt(prompt)

            prev_lines.append(line)

            try:
                result = json.loads(response_text)
            except Exception:
                logger.warning(
                    f"Could not parse response. Defaulting to non-ambiguous.\n"
                    f"Response: {response_text}"
                )
                result = {
                    "has_important_mistranscription": False
                }

            has_mistranscription = result.get(
                "has_important_mistranscription",
                False
            )

            if has_mistranscription:
                markdown_chunks.append(
                    f"**{line}**"
                )
            else:
                markdown_chunks.append(
                    line
                )

        with open(output_preds_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(markdown_chunks))


    