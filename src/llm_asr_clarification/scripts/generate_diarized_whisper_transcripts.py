import os
import argparse
from llm_asr_clarification import get_logger
import whisper
import torch
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F

# ┌───────────────────────────────────────────────┐
# │                 HELPER METHODS                │
# └───────────────────────────────────────────────┘
def extract_enrollment_embedding(audio_path, classifier, device):
    """
    Loads an individual headset file and extracts a 30-second 
    voice print to act as the reference embedding for this speaker.
    """
    audio_np = whisper.load_audio(audio_path.as_posix())
    waveform = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    
    # Take a 30-second slice starting at the 1-minute mark 
    # to avoid the initial silence of the meeting setup
    start_frame = 16000 * 60 
    end_frame = start_frame + (16000 * 30)
    
    # Fallback if the file is very short
    if waveform.shape[1] < end_frame:
        chunk = waveform
    else:
        chunk = waveform[:, start_frame:end_frame]
        
    with torch.no_grad():
        emb = classifier.encode_batch(chunk)
        
    # Squeeze out the batch dimensions so it's a flat vector
    return emb.squeeze()


# Wrap logging with tqdm
with logging_redirect_tqdm():

    # Driver Code
    def run(args_list=None):
        exp_name = os.path.basename(__file__)
        
        # Perform CLI Argument Parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--whisper-size", type=str, default="tiny")
        parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus")
        
        args, _ = parser.parse_known_args(args_list)

        # Parse CLI arguments to global variables
        WHISPER_SIZE = args.whisper_size
        DATASET_PATH = Path(args.dataset_path)

        # Other Global Variables
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        
        # Log important variables
        logger.info(f"Target device for model: {DEVICE}")

        # ┌───────────────────────────────────────────────┐
        # │                  LOAD MODELS                  │
        # └───────────────────────────────────────────────┘
        logger.info("Loading Whisper Model...")
        model = whisper.load_model(WHISPER_SIZE).to(DEVICE)

        logger.info("Loading SpeechBrain ECAPA-TDNN Model...")
        speaker_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE}
        )

        # ┌───────────────────────────────────────────────┐
        # │                   LOAD DATA                   │
        # └───────────────────────────────────────────────┘
        # Fetch all dataset meeting folders
        meeting_folders = [f for f in DATASET_PATH.iterdir() 
                           if (f.is_dir() and 
                               f.name not in ["ami_public_manual_1.6.2", "xinlu_data"])]

        # Process all Meeting Folders
        for meeting_folder in tqdm(meeting_folders, desc="Processing Meetings"):

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
                    DEVICE
                )

            # ┌───────────────────────────────────────────────┐
            # │                 TRANSCRIPTION                 │
            # └───────────────────────────────────────────────┘
            logger.info("Running Whisper Transcription on the Mixed Audio...")
            audio_np = whisper.load_audio(mix_file_path.as_posix())
            waveform = torch.from_numpy(audio_np).unsqueeze(0).to(DEVICE)
            
            result = model.transcribe(audio=audio_np)
            whisper_segments = result["segments"]


            # ┌───────────────────────────────────────────────┐
            # │          CLASSIFY SEGMENTS VIA ECAPA          │
            # └───────────────────────────────────────────────┘
            logger.info("Classifying speaker for each transcribed segment...")
            speaker_separated_data = []

            for segment in whisper_segments:
                seg_start = segment["start"]
                seg_end = segment["end"]
                text = segment["text"].strip()
                
                start_frame = int(seg_start * 16000)
                end_frame = int(seg_end * 16000)
                chunk_tensor = waveform[:, start_frame:end_frame]
                
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

            transcript_file_path = os.path.join(transcripts_folder, f"whisper_{WHISPER_SIZE}_ecapa_transcript.txt")
            with open(transcript_file_path, "w", encoding="utf-8") as f:
                f.write("".join(diarized_lines))
                
            logger.info(f"Saved transcript for {transcript_file_path}")

            break

