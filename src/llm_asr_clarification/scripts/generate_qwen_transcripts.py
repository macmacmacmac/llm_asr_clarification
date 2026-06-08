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
        parser.add_argument("--qwen-model-name", type=str, default="Qwen/Qwen3-ASR-1.7B")
        parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")

        args, _ = parser.parse_known_args(args_list)

        # Parse CLI arguments to global variables
        QWEN_MODEL_NAME = args.qwen_model_name
        DATASET_PATH = Path(args.dataset_path)

        # Other Global Variables
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        TARGET_SAMPLING_RATE = 16000
        
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


        # Load Voice Activity Detection (VAD) Model
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


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

            # Get Speech timestamps
            speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sample_rate)

            # Merge speech timestamps together into LLM-friednly chunks
            merged_timestamps = merge_timestamps(speech_timestamps, sample_rate)
            
            
            # ┌───────────────────────────────────────────────┐
            # |               TRANSCRIBE CHUNKS               │
            # └───────────────────────────────────────────────┘
            full_transcript = []

            for segment in tqdm(merged_timestamps, desc="Transcribing VAD chunks"):
                
                # Slice waveform
                chunk = waveform[segment['start']: segment['end']]

                # Write to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_file_path = temp_wav.name
                sf.write(temp_file_path, chunk, sample_rate)

                # Transcribe this specific chunk
                # TODO: Look into sending all chunks in one go as transcribe returns a List
                result = model.transcribe(audio=temp_file_path, language="English")[0]

                if hasattr(result, "text") and result.text.strip():
                        full_transcript.append(result.text.strip())

                # Clean up the temp file
                if os.path.exists(temp_file_path):
                     os.remove(temp_file_path)
            
            # Combine all chunk transcripts
            combined_text = " ".join(full_transcript)
            




            # ipdb.set_trace()

            # ┌───────────────────────────────────────────────┐
            # │                SAVE TRANSCRIPT                │
            # └───────────────────────────────────────────────┘
            transcript_file_path = transcripts_folder / "qwen_transcript.txt"
            with open(transcript_file_path, "w") as f:
                f.write(combined_text)
                
            logger.info(f"Saved transcript for {transcript_file_path}")

