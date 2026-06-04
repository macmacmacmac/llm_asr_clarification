import os
import argparse
from llm_asr_clarification import get_logger, OpenAIWrapper
from llm_asr_clarification.models.prompts import AMBIGUITY_PROMPT
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
    parser.add_argument("--msg", type=str, default="example")
    # parser.add_argument("--ami_path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--ami_path", type=str, default="/group/jrwhitehill/amicorpus")
    parser.add_argument("--txt_file_to_do", type=str, default="large_transcript")
    parser.add_argument("--meeting_to_do", type=str, default="/group/jrwhitehill/amicorpus/ES2005d")

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

    def split_into_sentences(text: str):
        """
        Basic sentence splitter.
        You can replace with nltk if desired.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]


    def chunk_sentences(sentences, chunk_size=5):
        for i in range(0, len(sentences), chunk_size):
            yield sentences[i:i + chunk_size]

    # directories of meetings
    if args.meeting_to_do:
        meeting_paths = [args.meeting_to_do]
    else:
        meeting_paths = [entry.path for entry in os.scandir(args.ami_path) if 'ami_public_manual_1.6.2' not in entry.name]
    for meeting_path in tqdm(meeting_paths):
        file_todo_path = os.path.join(meeting_path, "transcripts", f"{args.txt_file_to_do}.txt")
        output_preds_path = os.path.join(meeting_path, "transcripts", f"{args.txt_file_to_do}_ambiguity_preds.txt")
        
        logger.info(f"I am doing this file: {file_todo_path}")
        
        chatgpt = OpenAIWrapper()

        # TODO: open the text file at file_todo_path
        # TODO: iterate through text chunks in chunks of let's say 5 sentences, 
        #      run it through the ambiguity detection prompt
        #       
        #       this should return a dict-like object
        #       {
        #           "has_material_mistranscription": boolean
        #       }
        #       If that text chunk has mistranscription, it should be bolded

        # Read transcript
        with open(file_todo_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        sentences = split_into_sentences(transcript_text)

        markdown_chunks = []

        for sentence_chunk in chunk_sentences(sentences, chunk_size=5):

            transcript_excerpt = " ".join(sentence_chunk)

            prompt = AMBIGUITY_PROMPT.format(
                transcript_excerpt=transcript_excerpt
            )

            response_text = chatgpt.prompt_chatgpt(prompt)

            try:
                result = json.loads(response_text)
            except Exception:
                logger.warning(
                    f"Could not parse response. Defaulting to non-ambiguous.\n"
                    f"Response: {response_text}"
                )
                result = {
                    "has_material_mistranscription": False
                }

            has_mistranscription = result.get(
                "has_material_mistranscription",
                False
            )

            if has_mistranscription:
                markdown_chunks.append(
                    f"**{transcript_excerpt}**"
                )
            else:
                markdown_chunks.append(
                    transcript_excerpt
                )

        with open(output_preds_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(markdown_chunks))


    