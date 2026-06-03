import os
import argparse
from llm_asr_clarification import get_logger
import whisper
from pathlib import Path
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Wrap logging with tqdm
with logging_redirect_tqdm():

    # Driver Code
    def run(args_list=None):
        exp_name = os.path.basename(__file__)
        
        # Perform CLI Argument Parsing=================================================
        parser = argparse.ArgumentParser()
        parser.add_argument("--whisper-size", type=str, default="tiny")
        parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")

        args, _ = parser.parse_known_args(args_list)

        # Parse CLI arguments to global variables
        WHISPER_SIZE = args.whisper_size
        DATASET_PATH = Path(args.dataset_path)

        # Other Global Variables
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build the logger here
        # first arg is
        logger = get_logger(exp_name)    
        logger.info(
            f"{"="*100}\n\t\t\t\tRunning script: {exp_name}\n{"="*100}"
        )

        # Log received args
        received_args_log = ""
        for arg, value in vars(args).items():
            received_args_log += f"|---> {arg}: {value}\n"
        logger.info(
            f"Received the following arguments:\n{received_args_log}"
        )
        
        # Log important variables
        logger.info(f"Target device for model: {DEVICE}")

        # Load Whisper Model
        model = whisper.load_model(WHISPER_SIZE).to(DEVICE)

        # Fetch all dataset meeting folders
        meeting_folders = [meeting_folder for meeting_folder in DATASET_PATH.iterdir() if (meeting_folder.is_dir() and meeting_folder.name != "ami_public_manual_1.6.2")]
        for meeting_folder in tqdm(meeting_folders):

            # Prep audio and transcript folders
            audio_folder = meeting_folder / "audio"
            transcripts_folder = meeting_folder / "transcripts"

            # Fetch the audio file path and transcribe it
            audio_file_path = [wav_file for wav_file in audio_folder.rglob("*.wav")][0]
            result = model.transcribe(audio_file_path.as_posix())

            # Write the transcription to a file
            transcript_file_path = os.path.join(transcripts_folder, f"{WHISPER_SIZE}_transcript.txt")
            with open(transcript_file_path, "w") as f:
                f.write(result["text"])
            logger.info(f"Saved transcript for {transcript_file_path}")

