import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from jiwer import wer
import jiwer
import pandas as pd

# ┌───────────────────────────────────────────────┐
# │                HELPER_METHODS                 │
# └───────────────────────────────────────────────┘
def read_and_flatten(file_path: str) -> str:
    """Reads a file and converts all sentences/lines into one massive clean string."""
    with open(file_path, 'r') as f:
        # Read entire file, replace internal line breaks with spaces
        raw_text = f.read().replace('\n', ' ').replace('\r', ' ')
        return raw_text


# Driver Code
# Wrap logging with tqdm
with logging_redirect_tqdm():

    def run(args_list=None):

        # ┌───────────────────────────────────────────────┐
        # │                 HOUSEKEEPING                  │
        # └───────────────────────────────────────────────┘
        exp_name = os.path.basename(__file__)
        
        # Perform CLI Argument Parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")

        args, _ = parser.parse_known_args(args_list)

        # Parse CLI arguments to global variables
        DATASET_PATH = Path(args.dataset_path)

        # Other Global Variables
        TRANSCRIPT_FILE_NAMES = {
            "large_transcript.txt",
            "tiny_transcript.txt",
            "qwen_transcript.txt",
            "whisper-large-v3_transcript.txt",
            "whisper-tiny_transcript.txt"
            }

        # Define a robust transformation pipeline
        # This applies data cleaning steps in order, from top to bottom
        JIWER_TRANSFORM = jiwer.Compose([
            jiwer.ToLowerCase(),                # Convert all text to lowercase
            jiwer.RemovePunctuation(),          # Strip characters like commas, periods, question marks
            jiwer.RemoveMultipleSpaces(),       # Turn multi-spaces into a single space
            jiwer.Strip(),                      # Clean up leading/trailing whitespaces
            jiwer.ReduceToListOfListOfWords()   # Format text tokens perfectly for jiwer's internal engine
        ])
        
        
        
        # Build the logger here
        # first arg is
        logger = get_logger(exp_name)    
        logger.info(
            f"{"="*100}\n\t\t\t\tRunning script: {exp_name}\n{"="*100}"
        )

        # Log received args
        received_args_log = "".join([f"|---> {arg}: {value}\n" for arg, value in vars(args).items()])
        logger.info(
            f"Received the following arguments:\n{received_args_log}"
        )

        # ┌───────────────────────────────────────────────┐
        # │                  LOAD DATA                    │
        # └───────────────────────────────────────────────┘
        # Fetch all transcript folders in the dataset
        transcript_folders = [item / "transcripts" 
                              for item in DATASET_PATH.iterdir() 
                              if (item.is_dir() and 
                                  item.name != "ami_public_manual_1.6.2" and 
                                  item.name != "xinlu_data")]


        # ┌───────────────────────────────────────────────┐
        # │          PERFORM INDIVIDUAL SCORING           │
        # └───────────────────────────────────────────────┘
        all_scores = []
        for transcript_folder in tqdm(transcript_folders, "Generating scores"):
            scores = {"meeting_name": f"{os.path.dirname(transcript_folder).split('/')[-1]}"}
            ground_truth_transcript_file = transcript_folder / "parsed_gt.txt"
            generated_transcript_files = [
                file for file in transcript_folder.iterdir()
                if file.is_file() and file.name in TRANSCRIPT_FILE_NAMES
            ]
            ground_truth = read_and_flatten(ground_truth_transcript_file)

            for file in generated_transcript_files:
                generated = read_and_flatten(file)
                wer_score = wer(
                    reference = ground_truth,
                    hypothesis = generated,
                    reference_transform= JIWER_TRANSFORM,
                    hypothesis_transform = JIWER_TRANSFORM
                ) * 100
                scores[file.name] = round(wer_score, 4)

            all_scores.append(scores)

        # ┌───────────────────────────────────────────────┐
        # │          PERFORM AGGREGATE SCORING            │
        # └───────────────────────────────────────────────┘
        df = pd.DataFrame(all_scores)

        for col in df.columns.to_list():
            if col != 'meeting_name':
                mean_wer = df[col].mean()
                logger.info(f"{col} WER: {mean_wer}")

        


        df.to_csv(DATASET_PATH / "final_wer_scores.csv", index=False)


