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
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F
from llm_asr_clarification.utils.diarization_utils import extract_enrollment_embedding
import whisper

# Completely mute all warnings
transformers.logging.set_verbosity_error()

SAMPLING_RATE = 16_000


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
    parser.add_argument("--whisper-size", type=str, default="tiny")
    parser.add_argument("--meeting-name", type=str, default="")

    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    MODEL_NAME = args.model_name
    DATASET_PATH = Path(args.dataset_path)
    WHISPER_SIZE = args.whisper_size

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
    if args.meeting_name:
        meeting_folders=[DATASET_PATH / args.meeting_name]
    else:
        # Fetch all dataset meeting folders
        meeting_folders = [f for f in DATASET_PATH.iterdir() 
                            if (f.is_dir() and 
                                f.name not in ["ami_public_manual_1.6.2", "xinlu_data"])]
    

    # ┌───────────────────────────────────────────────┐
    # │                 LOAD MODELS                   │
    # └───────────────────────────────────────────────┘
    logger.info(f"Loading HF Model: {MODEL_NAME}...")

    # AutoProcessor handles text tokenization AND audio feature extraction
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # AutoModelForSpeechSeq2Seq handles the actual encoder-decoder network
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Load Whisper Model for time segmentation
    logger.info("Loading Whisper Model (For Time segmentation)...")
    whisper_model = whisper.load_model(WHISPER_SIZE).to(DEVICE)

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

    # ┌───────────────────────────────────────────────┐
    # │               LOAD ASR PIPELINE               │
    # └───────────────────────────────────────────────┘
    ASR_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
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
                
                speaker_embedding = extract_enrollment_embedding(
                    headset_file, 
                    speaker_classifier, 
                    vad_model,
                    get_speech_timestamps,
                    DEVICE
                )

                if speaker_embedding is not None:
                    enrolled_profiles[speaker_id] = speaker_embedding


            # TODO: This section is a pragmatic solution and is only temporary
            # until we find a better solution for time segmentation
            # ┌───────────────────────────────────────────────┐
            # │        FETCH TIME SEGMENTS USING WHISPER      │
            # └───────────────────────────────────────────────┘
            logger.info("Running Time Segmentation using Whisper...")
            audio_np = whisper.load_audio(mix_file_path.as_posix()) # (num_frames, )
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(DEVICE) # (1, num_frames)
            speech_timestamps = whisper_model.transcribe(audio=audio_np)["segments"]

            # ┌───────────────────────────────────────────────┐
            # │          TRANSCRIPTION AND DIARIZATION        │
            # └───────────────────────────────────────────────┘
            logger.info(f"Transcribing audio: {mix_file_path.name}")
            waveform, sample_rate = sf.read(mix_file_path) # waveform shape: (num_frames,)
            waveform = waveform.astype("float32")

            # Pad by some milliseconds (in frames) for Whisper's acoustic context
            pad_frames = int(0.1 * SAMPLING_RATE)

            # Diarization variables
            speaker_separated_data = []
            last_valid_speaker = "UNKNOWN" # Keep track of the last successfully identified speaker
            
            for segment in tqdm(speech_timestamps, desc="Transcribing"):

                # Extract Start and End Frame
                start_frame = int(segment["start"] * SAMPLING_RATE)
                end_frame = int(segment["end"] * SAMPLING_RATE)

                # Apply Padding
                start_frame = max(0, start_frame - pad_frames)
                end_frame = min(len(waveform), end_frame + pad_frames)

                # Retrieve audio chunk
                chunk = waveform[start_frame: end_frame]

                # TRANSCRIPTION
                transcription = ASR_PIPELINE(chunk)
                text = transcription["text"].strip()

                # Skip further processing if no text was transcribed
                if not text:
                    continue

                # DIARIZATION
                chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE) # (1, num_frames)

                # If the chunk contains audio less than 1 second
                # Assign last speaker as the speaker
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
                
                speaker_separated_data.append({
                    "speaker": best_speaker,
                    "text": text,
                    "start": int(segment["start"]),
                    "end": int(segment["end"])
                })

            # ┌───────────────────────────────────────────────┐
            # │                FORMAT & SAVE                  │
            # └───────────────────────────────────────────────┘
            diarized_lines = []
            if speaker_separated_data:

                # Use the first data as the initial point
                last_speaker = speaker_separated_data[0]["speaker"]
                combined_text = speaker_separated_data[0]["text"]
                start = speaker_separated_data[0]["start"]
                end = speaker_separated_data[0]["end"]


                for data in speaker_separated_data[1:]:
                    # If the speaker hasn't changed
                    if data["speaker"] == last_speaker:
                        # Combine the texts
                        combined_text += " " + data["text"]

                        # Merge the timestamps
                        end = data["end"]
                    
                    # The speaker has changed
                    else:
                        diarized_lines.append(f"({start} - {end})[{last_speaker}]: {combined_text.strip()}\n")
                        last_speaker = data["speaker"]
                        combined_text = data["text"]
                        start = data["start"]
                        end = data["end"]

                diarized_lines.append(f"({start} - {end})[{last_speaker}]: {combined_text.strip()}\n")

            transcript_file_path = transcripts_folder / f"custom_{MODEL_NAME.split('/')[-1]}_transcript.txt"
            with open(transcript_file_path, "w", encoding="utf-8") as f:
                f.write("".join(diarized_lines))
                
            logger.info(f"Saved transcript for {transcript_file_path}\n\n")
            

