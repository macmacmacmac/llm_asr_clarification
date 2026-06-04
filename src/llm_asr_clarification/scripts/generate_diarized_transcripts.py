import os
import argparse
from llm_asr_clarification import get_logger
import whisper
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import warnings
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Suppress the broken torchcodec import warning from pyannote
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")

# Resolve the Pyannote TF32 reproducibility warning by explicitly configuring torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load all env variables
load_dotenv()

# Wrap logging with tqdm
with logging_redirect_tqdm():

    # Driver Code
    def run(args_list=None):
        exp_name = os.path.basename(__file__)
        
        # Perform CLI Argument Parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--whisper-size", type=str, default="tiny")
        parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")
        # parser.add_argument("--audio-path", type=str, default="./datasets/amicorpus/ES2005a/audio/ES2005a.Mix-Headset.wav")

        args, _ = parser.parse_known_args(args_list)

        # Parse CLI arguments to global variables
        WHISPER_SIZE = args.whisper_size
        # AUDIO_PATH = args.audio_path
        DATASET_PATH = Path(args.dataset_path)

        # Other Global Variables
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        HF_TOKEN = os.getenv("HF_TOKEN")
        
        # Init Logger
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
        # │                  LOAD MODELS                  │
        # └───────────────────────────────────────────────┘
        logger.info("Loading Whisper Model...")
        model = whisper.load_model(WHISPER_SIZE).to(DEVICE)
        
        logger.info("Loading Pyannote Diarization Pipeline...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        # Pyannote handles device placement slightly differently
        diarization_pipeline.to(torch.device(DEVICE))


        # Fetch all dataset meeting folders
        meeting_folders = [meeting_folder for meeting_folder in DATASET_PATH.iterdir() if (meeting_folder.is_dir() and meeting_folder.name != "ami_public_manual_1.6.2")]

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders):

            # Prep audio and transcript folders
            audio_folder = meeting_folder / "audio"
            transcripts_folder = meeting_folder / "transcripts"

            # Fetch the audio file path and transcribe it
            audio_file_path = [wav_file for wav_file in audio_folder.rglob("*.wav")][0]

            # ┌───────────────────────────────────────────────┐
            # │     PERFORM DIARIZATION AND TRANSCRIPTION     │
            # └───────────────────────────────────────────────┘
            logger.info("Loading audio into memory using Whisper's ffmpeg utility...")
                
            # This loads the audio as a float32 mono numpy array sampled at whisper.audio.SAMPLE_RATE
            audio_np = whisper.load_audio(audio_file_path.as_posix())
            
            # Convert to a PyTorch Tensor and add a channel dimension: shape (1, num_samples)
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
            
            # Create the dictionary that Pyannote accepts
            audio_in_memory = {
                "waveform": waveform, 
                "sample_rate": whisper.audio.SAMPLE_RATE
            }

            logger.info("Running Pyannote Diarization...")

            # Pass the dictionary INSTEAD of the AUDIO_PATH string
            diarization = diarization_pipeline(
                audio_in_memory,
                num_speakers = 4
            )

            logger.info("Running Whisper Transcription...")

            # Pass the pre-loaded numpy array to prevent Whisper from re-reading the file
            result = model.transcribe(audio=audio_np)
            whisper_segments = result["segments"]
            


            # ┌───────────────────────────────────────────────┐
            # │        ALIGN SPEAKERS WITH TRANSCRIPTS        │
            # └───────────────────────────────────────────────┘
            logger.info("Aligning transcripts with speakers...")
            speaker_separated_data = []

            for segment in whisper_segments:
                seg_start = segment["start"]
                seg_end = segment["end"]
                text = segment["text"].strip()
                
                # Find the speaker with the maximum overlap for this segment
                max_overlap = 0
                best_speaker = "UNKNOWN_SPEAKER"

                # ipdb.set_trace()
                
                for turn, speaker_literal, speaker in diarization.speaker_diarization.itertracks(yield_label=True):

                    # ipdb.set_trace()

                    # Calculate the intersection of the two time windows
                    overlap = max(0, min(seg_end, turn.end) - max(seg_start, turn.start))
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = speaker
                
                # Format the output line
                # line = f"[{seg_start:05.2f} - {seg_end:05.2f}] {best_speaker}: {text}"
                # line = f"[{best_speaker}]: {text}\n"

                speaker_separated_data.append((best_speaker, text))


            # Merge speakers
            diarized_lines = []
            last_speaker = speaker_separated_data[0][0]
            combined_text = speaker_separated_data[0][1]
            for i in range(1, len(speaker_separated_data)):
                current_speaker = speaker_separated_data[i][0]
                text = speaker_separated_data[i][1]

                if current_speaker == last_speaker:
                    combined_text += text
                else:
                    # Save last speaker's stuff
                    diarized_lines.append(f"[{last_speaker}]: {combined_text}\n")

                    # Make current speaker as last speaker
                    last_speaker = current_speaker
                    combined_text = text

            diarized_lines.append(f"[{last_speaker}]: {combined_text}\n")
            

            # ┌───────────────────────────────────────────────┐
            # │                  SAVE OUTPUT                  │
            # └───────────────────────────────────────────────┘
            transcript_file_path = os.path.join(transcripts_folder, f"whisper_{WHISPER_SIZE}_diarized_transcript.txt")
            with open(transcript_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(diarized_lines))
                
            logger.info(f"Saved transcript for {transcript_file_path}")

