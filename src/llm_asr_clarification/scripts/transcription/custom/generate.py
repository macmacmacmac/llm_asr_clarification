import os
import argparse
from llm_asr_clarification import get_logger
from pathlib import Path
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import soundfile as sf
import ipdb
import transformers
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torchaudio.functional as F_audio

# Completely mute all warnings
transformers.logging.set_verbosity_error()

# ┌───────────────────────────────────────────────┐
# │                HELPER METHODS                 │
# └───────────────────────────────────────────────┘
def merge_timestamps(timestamps, sample_rate, max_chunk_len_sec=30.0, buffer_sec=0.5):
    """
    Merges tiny VAD chunks into larger blocks and adds safety padding.
    """
    if not timestamps:
        return []

    merged = []
    current_start = timestamps[0]['start']
    current_end = timestamps[0]['end']
    
    # Convert seconds to frames
    max_frames = int(max_chunk_len_sec * sample_rate)
    buffer_frames = int(buffer_sec * sample_rate)

    for i in range(1, len(timestamps)):
        next_start = timestamps[i]['start']
        next_end = timestamps[i]['end']
        
        # Calculate how long the chunk would be if we merged them
        proposed_len = next_end - current_start
        
        # If merging them keeps it under our maximum allowed size, merge them!
        if proposed_len <= max_frames:
            current_end = next_end
        else:
            # The chunk is big enough. Add padding and save it.
            merged.append({
                'start': max(0, current_start - buffer_frames), 
                'end': current_end + buffer_frames
            })
            # Start a new chunk
            current_start = next_start
            current_end = next_end
            
    # Append the final chunk
    merged.append({
        'start': max(0, current_start - buffer_frames), 
        'end': current_end + buffer_frames
    })
    
    return merged

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
        parser.add_argument("--model-name", type=str, default="openai/whisper-tiny")
        parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")

        args, _ = parser.parse_known_args(args_list)

        # Parse CLI arguments to global variables
        MODEL_NAME = args.model_name
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
        received_args_log = "".join([f"|---> {arg}: {value}\n" for arg, value in vars(args).items()])
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
        logger.info(f"Loading Model: {MODEL_NAME}...")

        # AutoProcessor handles text tokenization AND audio feature extraction
        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        # AutoModelForSpeechSeq2Seq handles the actual encoder-decoder network
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            low_cpu_mem_usage=True
        )

        # Determine target sampling rate 
        TARGET_SAMPLING_RATE = 16000

        # Load Voice Activity Detection (VAD) Model
        logger.info("Loading Silero VAD Model...")
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (get_speech_timestamps, _, _, _, _) = utils


        for meeting_folder in tqdm(meeting_folders):

            # Prep audio and transcript folders
            audio_folder = meeting_folder / "audio"
            transcripts_folder = meeting_folder / "transcripts"

            # Fetch the audio file path
            audio_file_path = list(audio_folder.rglob("*.wav"))[0]

            # ┌───────────────────────────────────────────────┐
            # │           AUDIO CHUNKING USING VAD            │
            # └───────────────────────────────────────────────┘
            logger.info(f"Processing audio: {audio_file_path.name}")
            waveform, sample_rate = sf.read(audio_file_path) # waveform shape: (num_frames, num_channels)

            # If stereo (shape [frames, channels]), convert to mono by averaging channels
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)

            # Convert numpy array to torch tensor for Silero VAD
            wav_tensor = torch.from_numpy(waveform).float()

            # Robust Resampling: Ensure audio matches the model's required sample rate
            if sample_rate != TARGET_SAMPLING_RATE:
                logger.info(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLING_RATE}Hz...")
                wav_tensor = F_audio.resample(wav_tensor, sample_rate, TARGET_SAMPLING_RATE)
                
                # Update the waveform array and sample rate tracker
                waveform = wav_tensor.numpy()
                sample_rate = TARGET_SAMPLING_RATE

            # Get Speech timestamps
            speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sample_rate)

            # Merge speech timestamps together into LLM-friendly chunks
            merged_timestamps = merge_timestamps(speech_timestamps, sample_rate)
            
            
            # ┌───────────────────────────────────────────────┐
            # |               TRANSCRIBE CHUNKS               │
            # └───────────────────────────────────────────────┘
            full_transcript = []

            for segment in tqdm(merged_timestamps, desc="Transcribing chunks"):
                
                # Slice waveform
                chunk = waveform[segment['start']: segment['end']]

                # Process chunk into model-specific tensor features (e.g. log-mel spectrogram)
                inputs = processor(
                    chunk, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt",
                ).to(DEVICE)

                # Cast float tensors to bfloat16 to match the model precision
                if "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
                
                # Generate Token IDs
                with torch.no_grad():
                    # Check if the model has a strict target position limit
                    max_allowed = getattr(model.config, "max_target_positions", 4096)
                    
                    # Leave a buffer of 20 tokens for hidden start/prompt tokens
                    safe_max_tokens = min(400, max_allowed - 20) 

                    generated_ids = model.generate(**inputs, max_new_tokens=safe_max_tokens)

                    # generated_ids = model.generate(**inputs, max_new_tokens=256)
                
                # Decode Token IDs back to text strings
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if transcription.strip():
                    full_transcript.append(transcription.strip())
            
            # Combine all chunk transcripts
            combined_text = " ".join(full_transcript)
            

            # ┌───────────────────────────────────────────────┐
            # │                SAVE TRANSCRIPT                │
            # └───────────────────────────────────────────────┘
            transcript_file_path = transcripts_folder / f"custom_{MODEL_NAME.split('/')[-1]}_transcript.txt"
            with open(transcript_file_path, "w") as f:
                f.write(combined_text)
                
            logger.info(f"Saved transcript for {transcript_file_path}")

