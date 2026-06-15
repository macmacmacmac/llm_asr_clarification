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
    parser.add_argument("--meeting-name", type=str, default="")

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

            # ┌───────────────────────────────────────────────┐
            # │          TRANSCRIPTION AND DIARIZATION        │
            # └───────────────────────────────────────────────┘
            logger.info(f"Transcribing audio: {mix_file_path.name}")
            waveform, sample_rate = sf.read(mix_file_path) # waveform shape: (num_frames,)
            waveform = waveform.astype("float32")

            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1) # Flatten to mono

            # Convert numpy array to torch tensor for Silero VAD
            wav_tensor = torch.from_numpy(waveform).to(DEVICE)

            # Get Speech timestamps
            speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sample_rate)

            # Pad by 0.5 seconds (in frames) for Whisper's acoustic context
            pad_frames = int(0.5 * SAMPLING_RATE)

            # Diarization variables
            speaker_separated_data = []
            last_valid_speaker = "UNKNOWN" # Keep track of the last successfully identified speaker
            
            for segment in tqdm(speech_timestamps, desc="Transcribing"):
                vad_start = segment["start"]
                vad_end = segment["end"]

                # TRANSCRIPTION
                asr_start = max(0, vad_start - pad_frames)
                asr_end = min(len(waveform), vad_end + pad_frames)
                asr_chunk = waveform[asr_start: asr_end]
                transcription = ASR_PIPELINE(asr_chunk)
                text = transcription["text"].strip()

                # Skip further processing if no text was transcribed
                if not text:
                    continue

                # DIARIZATION
                ecapa_chunk = waveform[vad_start: vad_end]
                chunk_tensor = torch.from_numpy(ecapa_chunk).unsqueeze(0).to(DEVICE)

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
            

