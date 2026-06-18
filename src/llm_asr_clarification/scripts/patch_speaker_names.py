import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
import ipdb
import shutil
from llm_asr_clarification.utils.diarization_utils import get_headset_to_speaker_map


# Driver Code
def run(args_list=None):
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--seed", type=int, default=47)
    
    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    DATASET_PATH = Path(args.dataset_path)
    TRANSCRIPT_FILE_NAME = "whisper_tiny_diarized_transcript.txt"
    
    # Init Logger
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
    # Fetch all dataset meeting folders
    meeting_folders = [f for f in DATASET_PATH.iterdir() 
                        if (f.is_dir() and 
                            f.name not in ["ami_public_manual_1.6.2", "xinlu_data"])]


    for meeting_folder in meeting_folders:
        logger.info(f"Processing meeting {meeting_folder.name}")

        # Prep audio and transcript folders
        audio_folder = meeting_folder / "audio"
        transcripts_folder = meeting_folder / "transcripts"

        # Fetch all wav files
        all_wavs = list(audio_folder.rglob("*.wav"))
        
        # Separate the Mix from the individual Headsets
        headset_files = [f for f in all_wavs if "Mix-Headset" not in f.name and "Headset" in f.name]


        # Construct a map of headset to speaker name
        headset_to_speaker_map = get_headset_to_speaker_map(headset_files)


        # Read the transcript
        transcript_path = transcripts_folder / TRANSCRIPT_FILE_NAME
        with open(transcript_path, "r") as f:
            transcript = f.read()

        
        # Perform Speaker Replacements
        for headset_name, speaker_name in headset_to_speaker_map.items():
            # transcript = transcript.replace(speaker_name, headset_name)
            transcript = transcript.replace(headset_name, speaker_name)


        # Save the replaced transcript
        with open(transcript_path, "w") as f:
            f.write(transcript)

