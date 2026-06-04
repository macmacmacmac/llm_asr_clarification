import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from qwen_asr import Qwen3ASRModel
import soundfile as sf
import tempfile
import ipdb
import transformers

# Completely mute all warnings
transformers.logging.set_verbosity_error()

# Wrap logging with tqdm
with logging_redirect_tqdm():

    # Driver Code
    def run(args_list=None):

        # ┌───────────────────────────────────────────────┐
        # │                 HOUSEKEEPING                  │
        # └───────────────────────────────────────────────┘
        exp_name = os.path.basename(__file__)
        
        # Perform CLI Argument Parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--qwen-model-name", type=str, default="Qwen/Qwen3-ASR-1.7B")
        parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")

        args, _ = parser.parse_known_args(args_list)

        # Parse CLI arguments to global variables
        # WHISPER_SIZE = args.whisper_size
        QWEN_MODEL_NAME = args.qwen_model_name
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

        # ┌───────────────────────────────────────────────┐
        # │                  LOAD DATA                    │
        # └───────────────────────────────────────────────┘
        # Fetch all meeting folders in the dataset
        meeting_folders = [item for item in DATASET_PATH.iterdir() if (item.is_dir() and item.name != "ami_public_manual_1.6.2")]
        

        # ┌───────────────────────────────────────────────┐
        # │                  LOAD MODEL                   │
        # └───────────────────────────────────────────────┘
        # Load Qwen Model
        # TODO: Add Flash Attention later
        model = Qwen3ASRModel.from_pretrained(
            QWEN_MODEL_NAME,
            dtype=torch.bfloat16,
            device_map=DEVICE,
            max_new_tokens=4096
        )

        for meeting_folder in tqdm(meeting_folders):

            # Prep audio and transcript folders
            audio_folder = meeting_folder / "audio"
            transcripts_folder = meeting_folder / "transcripts"

            # Fetch the audio file path
            audio_file_path = list(audio_folder.rglob("*.wav"))[0]

            # ┌───────────────────────────────────────────────┐
            # │                 AUDIO CHUNKING                │
            # └───────────────────────────────────────────────┘
            logger.info(f"Processing audio: {audio_file_path.name}")
            waveform, sample_rate = sf.read(audio_file_path) # waveform shape: (num_frames, num_channels)
            
            # Variables for Chunking
            chunk_duration = 30  # 30 seconds
            chunk_frames = int(chunk_duration * sample_rate)
            total_frames = waveform.shape[0]

            # List to hold individual chunk transcripts
            full_transcript = []

            # Iterate through the audio in 30-second chunks
            for i in tqdm(range(0, total_frames, chunk_frames), desc="Transcribing chunks", leave=False):

                # Slice the waveform across the frame dimension to get the chunk 
                chunk = waveform[i:i + chunk_frames]
                
                # Write to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_file_path = temp_wav.name
                
                
                # Save the chunk temporarily
                sf.write(temp_file_path, chunk, sample_rate)
                
                # Transcribe this specific chunk
                # TODO: Look into sending all chunks in one go as transcribe returns a List
                result = model.transcribe(audio=temp_file_path)[0]
                
                if hasattr(result, "text"):
                    # Clean up formatting and append
                    full_transcript.append(result.text.strip())
            
            
                # Always clean up the temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            # Combine all chunk transcripts
            combined_text = " ".join(full_transcript)


            # ┌───────────────────────────────────────────────┐
            # │                SAVE TRANSCRIPT                │
            # └───────────────────────────────────────────────┘
            transcript_file_path = transcripts_folder / "qwen_transcript.txt"
            with open(transcript_file_path, "w") as f:
                f.write(combined_text)
                
            logger.info(f"Saved transcript for {transcript_file_path}")

