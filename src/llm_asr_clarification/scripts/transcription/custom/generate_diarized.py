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
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F

# Completely mute all warnings
transformers.logging.set_verbosity_error()

SAMPLING_RATE = 16_000

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


def extract_enrollment_embedding(
        audio_path, 
        classifier, 
        vad_model, 
        get_speech_timestamps, 
        device,
        target_duration_sec = 30.0
    ):
    """
    Extracts a reference embedding by using VAD (Voice Activity Detection) to find and concatenate 
    pure speech segments, ignoring all silence and background noise.
    """
    audio_np, sample_rate = sf.read(audio_path) # waveform shape: (num_frames,)
    waveform = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)

    # Get Speech Timestamps using VAD model
    wav_tensor = waveform.squeeze(0)
    speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate = SAMPLING_RATE)

    # Collect speech chunks until we hit target duration
    speech_chunks = []
    collected_frames = 0
    target_frames = int(target_duration_sec * SAMPLING_RATE)

    # Keep collecting speech chunks until we hit target frames
    for segment in speech_timestamps:
        start = segment["start"]
        end = segment["end"]
        chunk = waveform[:, start:end]
        speech_chunks.append(chunk)

        collected_frames += (end - start)
        if collected_frames >= target_frames:
            break

    # Concatenate speech chunks into a block of continuous speech
    combined_speech = torch.cat(speech_chunks, dim = 1)

    # Trim combined speech chunk so its always target_duration_sec long
    combined_speech = combined_speech[:, :target_frames]

    with torch.no_grad():
        emb = classifier.encode_batch(combined_speech)
    
    return emb.squeeze()


# Driver Code
def run(args_list=None):

    # ┌───────────────────────────────────────────────┐
    # │                 HOUSEKEEPING                  │
    # └───────────────────────────────────────────────┘
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="openai/whisper-tiny")
    parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")
    parser.add_argument("--meeting-path", type=str, default="./datasets/amicorpus/ES2005d")

    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    MODEL_NAME = args.model_name
    DATASET_PATH = Path(args.dataset_path)

    # Other Global Variables
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
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
    if args.meeting_path:
        meeting_folders=[Path(args.meeting_path)]
    else:
        # Fetch all dataset meeting folders
        meeting_folders = [f for f in DATASET_PATH.iterdir() 
                            if (f.is_dir() and 
                                f.name not in ["ami_public_manual_1.6.2", "xinlu_data"])]
    

    # ┌───────────────────────────────────────────────┐
    # │                 LOAD MODELS                   │
    # └───────────────────────────────────────────────┘
    logger.info(f"Loading Model: {MODEL_NAME}...")

    # AutoProcessor handles text tokenization AND audio feature extraction
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # AutoModelForSpeechSeq2Seq handles the actual encoder-decoder network
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Load Voice Activity Detection (VAD) Model
    logger.info("Loading Silero VAD Model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps, _, _, _, _) = utils
    vad_model = vad_model.to(DEVICE)

    # Load ECAPA-TDNN Model for Speaker Enrollment and Diarization
    logger.info("Loading SpeechBrain ECAPA-TDNN Model...")
    speaker_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE}
    )

    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[logger]):

        # Prcoess all Meeting Folders
        for meeting_folder in tqdm(meeting_folders):

            # Prep audio and transcript folders
            audio_folder = meeting_folder / "audio"
            transcripts_folder = meeting_folder / "transcripts"

            # Fetch all wav files
            all_wavs = list(audio_folder.rglob("*.wav"))

            # Separate the Mix from the individual Headsets
            mix_file_path = [f for f in all_wavs if "Mix-Headset" in f.name][0]
            headset_files = [f for f in all_wavs if "Mix-Headset" not in f.name and "Headset" in f.name]

            # ┌───────────────────────────────────────────────┐
            # │               SPEAKER ENROLLMENT              │
            # └───────────────────────────────────────────────┘
            logger.info(f"Extracting enrollment embeddings for {len(headset_files)} speakers...")
            enrolled_profiles = {}
            
            for headset_file in headset_files:
                # Use the filename (e.g., "ES2005a.Headset-0") as the speaker label
                speaker_id = headset_file.stem.split('.')[-1] 
                
                enrolled_profiles[speaker_id] = extract_enrollment_embedding(
                    headset_file, 
                    speaker_classifier, 
                    vad_model,
                    get_speech_timestamps,
                    DEVICE
                )

            # ┌───────────────────────────────────────────────┐
            # │                 TRANSCRIPTION                 │
            # └───────────────────────────────────────────────┘
            logger.info(f"Processing audio: {mix_file_path.name}")
            waveform, sample_rate = sf.read(mix_file_path) # waveform shape: (num_frames,)

            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1) # Flatten to mono

            # Convert numpy array to torch tensor for Silero VAD
            wav_tensor = torch.from_numpy(waveform).to(DEVICE).float()

            # Get Speech timestamps
            speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sample_rate)

            # Merge speech timestamps together into LLM-friendly chunks
            # merged_timestamps = merge_timestamps(speech_timestamps, sample_rate)
            
            # Transcribe Chunks
            transcription_segments = []
            # for segment in tqdm(merged_timestamps, desc="Transcribing chunks"):
            for segment in tqdm(speech_timestamps, desc="Transcribing chunks"):
                
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
                    transcription_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": transcription.strip()
                    })
            
            
            # ┌───────────────────────────────────────────────┐
            # │          CLASSIFY SEGMENTS VIA ECAPA          │
            # └───────────────────────────────────────────────┘
            logger.info("Classifying speaker for each transcribed segment...")
            speaker_separated_data = []

            # Keep track of the last successfully identified speaker
            last_valid_speaker = "UNKNOWN"

            for segment in transcription_segments:
                start_frame = segment["start"]
                end_frame = segment["end"]
                text = segment["text"].strip()

                # Get the current chunk based on start and end frame
                chunk_array = waveform[start_frame: end_frame]

                # Convert to a 2D PyTorch Tensor and send to the GPU
                chunk_tensor = torch.from_numpy(chunk_array).float().unsqueeze(0).to(DEVICE)

                if chunk_tensor.shape[1] < SAMPLING_RATE:
                    best_speaker = last_valid_speaker
                
                else:

                    with torch.no_grad():
                        chunk_emb = speaker_classifier.encode_batch(chunk_tensor).squeeze()
                    
                    best_speaker = "UNKNOWN"
                    highest_sim = -1.0
                    
                    # Compare this phrase against all enrolled profiles using Cosine Similarity
                    for speaker_name, profile_emb in enrolled_profiles.items():
                        sim = F.cosine_similarity(chunk_emb, profile_emb, dim=0).item()
                        if sim > highest_sim:
                            highest_sim = sim
                            best_speaker = speaker_name

                    if best_speaker != "UNKNOWN":
                        last_valid_speaker = best_speaker
                
                speaker_separated_data.append((best_speaker, text))

            # ┌───────────────────────────────────────────────┐
            # │                FORMAT & SAVE                  │
            # └───────────────────────────────────────────────┘
            diarized_lines = []
            if speaker_separated_data:
                last_speaker, combined_text = speaker_separated_data[0]
                
                for current_speaker, text in speaker_separated_data[1:]:
                    if current_speaker == last_speaker:
                        combined_text += " " + text
                    else:
                        diarized_lines.append(f"[{last_speaker}]: {combined_text.strip()}\n")
                        last_speaker = current_speaker
                        combined_text = text

                diarized_lines.append(f"[{last_speaker}]: {combined_text.strip()}\n")

            transcript_file_path = transcripts_folder / f"custom_{MODEL_NAME.split('/')[-1]}_transcript.txt"
            with open(transcript_file_path, "w", encoding="utf-8") as f:
                f.write("".join(diarized_lines))
                
            logger.info(f"Saved transcript for {transcript_file_path}\n\n")
            

