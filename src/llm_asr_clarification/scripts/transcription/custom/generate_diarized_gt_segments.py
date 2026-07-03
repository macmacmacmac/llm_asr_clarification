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
from llm_asr_clarification.utils.diarization_utils import (
    extract_enrollment_embedding, get_headset_to_speaker_map
)
from llm_asr_clarification.utils import extract_timestamps
import json

# Completely mute all warnings
transformers.logging.set_verbosity_error()


# Global Variables
SAMPLING_RATE = 16_000


def perform_transcription(
        model,
        processor,
        audio_chunk,
        num_beams = 3,
        print_beam_results = False
):
    # Preprocess the audio into spectrogram features
    inputs = processor(
        audio_chunk, 
        sampling_rate=SAMPLING_RATE, 
        return_tensors="pt"
    ).to(DEVICE)

    # Cast features to match model precision if using bfloat16/half
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)

    # Generate text using the model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            temperature = 0.1,
            do_sample = True,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Decode Token IDs back to text strings
    decoded_results = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    
    # Calculate average log probabilities using transition scores
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    
    # Handle potential -inf values gracefully to prevent NaN when calculating mean
    transition_scores = torch.where(
        torch.isinf(transition_scores),
        torch.tensor(-100.0, device=transition_scores.device),
        transition_scores
    )
    scores = transition_scores.mean(dim=1).tolist()

    # TODO: Compute LLM Log Probs
    
    # Format the results
    beam_results = {}
    for i, (text, score) in enumerate(zip(decoded_results, scores)):
        beam_results[f'beam_{i+1}'] = {'text': text, 'asr_avg_log_prob': score}

        if print_beam_results:
            LOGGER.info(f"Beam {i + 1}: '{text}' (log prob: {score:.4f})")

    return beam_results


def get_ground_truth_segments(gt_transcript_path: Path):
    # Fetch all lines
    with open(gt_transcript_path, "r") as f:
        gt_lines = f.read().strip().split("\n")

    for i, line in enumerate(gt_lines):
        start, end = extract_timestamps(line)
        text = line.split(":")[1]
        gt_lines[i] = {
            "start": start,
            "end": end,
            "text": text
        }

    return gt_lines





# Driver Code
def run(args_list=None):

    # ┌───────────────────────────────────────────────┐
    # │                 HOUSEKEEPING                  │
    # └───────────────────────────────────────────────┘
    exp_name = os.path.basename(__file__)
    
    # Perform CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="openai/whisper-tiny")
    parser.add_argument("--dataset-path", type=str, default="./datasets/amicorpus/train")
    parser.add_argument("--whisper-size", type=str, default="tiny")
    parser.add_argument("--meeting-name", type=str, default="")

    args, _ = parser.parse_known_args(args_list)

    # Parse CLI arguments to global variables
    MODEL_NAME = args.model_name
    DATASET_PATH = Path(args.dataset_path)
    WHISPER_SIZE = args.whisper_size

    # Other Global Variables
    global DEVICE
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Build the LOGGER here
    # first arg is
    global LOGGER
    LOGGER = get_logger(exp_name)    
    LOGGER.info(
        f"{"="*100}\n\t\t\t\tRunning script: {exp_name}\n{"="*100}"
    )

    # Log received args
    received_args_log = "".join([f"|---> {arg}: {value}\n" for arg, value in vars(args).items()])
    LOGGER.info(
        f"Received the following arguments:\n{received_args_log}"
    )
    
    # Log important variables
    LOGGER.info(f"Target device for model: {DEVICE}")

    # ┌───────────────────────────────────────────────┐
    # │                  LOAD DATA                    │
    # └───────────────────────────────────────────────┘
    if args.meeting_name:
        meeting_folders=[DATASET_PATH / args.meeting_name]
    else:
        # Fetch all dataset meeting folders
        meeting_folders = [f for f in DATASET_PATH.iterdir() if f.is_dir() ]
    

    # ┌───────────────────────────────────────────────┐
    # │                 LOAD MODELS                   │
    # └───────────────────────────────────────────────┘
    LOGGER.info(f"Loading HF Model: {MODEL_NAME}...")

    # AutoProcessor handles text tokenization AND audio feature extraction
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # AutoModelForSpeechSeq2Seq handles the actual encoder-decoder network
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    # Load Voice Activity Detection (VAD) Model
    LOGGER.info("Loading Silero VAD Model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps, _, _, _, _) = utils
    vad_model = vad_model.to(DEVICE)

    # Load ECAPA-TDNN Model for Speaker Enrollment and Diarization
    LOGGER.info("Loading SpeechBrain ECAPA-TDNN Model...")
    speaker_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE}
    )

    # Wrap logging with tqdm
    with logging_redirect_tqdm(loggers=[LOGGER]):

        # Prcoess all Meeting Folders
        for meeting_folder in tqdm(meeting_folders):

            # Prep audio and transcript folders
            audio_folder = meeting_folder / "audio"
            transcripts_folder = meeting_folder / "transcripts"

            # Load Ground Truth Segments
            gt_segments = get_ground_truth_segments(transcripts_folder / "parsed_diarized_gt.txt")

            # Fetch all wav files
            all_wavs = list(audio_folder.rglob("*.wav"))

            # Separate the Mix from the individual Headsets
            mix_file_path = [f for f in all_wavs if "Mix-Headset" in f.name][0]
            headset_files = [f for f in all_wavs if "Mix-Headset" not in f.name and "Headset" in f.name]

            # ┌───────────────────────────────────────────────┐
            # │               SPEAKER ENROLLMENT              │
            # └───────────────────────────────────────────────┘
            LOGGER.info(f"Extracting enrollment embeddings for {len(headset_files)} speakers...")
            enrolled_profiles = {}

            # Construct a map of headset to speaker name
            headset_to_speaker_map = get_headset_to_speaker_map(headset_files)
            
            for headset_file in headset_files:
                # Use the filename (e.g., "ES2005a.Headset-0") as the speaker label
                speaker_id = headset_to_speaker_map[headset_file.stem.split('.')[-1]]
                
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
            LOGGER.info(f"Transcribing audio: {mix_file_path.name}")
            waveform, sample_rate = sf.read(mix_file_path) # waveform shape: (num_frames,)

            # If stereo (shape [frames, channels]), convert to mono by averaging channels
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)

            waveform = waveform.astype("float32")

            # Diarization variables
            speaker_separated_data = []
            last_valid_speaker = "UNKNOWN" # Keep track of the last successfully identified speaker
            
            for segment in tqdm(gt_segments, desc="Transcribing"):

                # Extract Start and End Frame
                start_frame = int(segment["start"] * SAMPLING_RATE)
                end_frame = int(segment["end"] * SAMPLING_RATE)

                # Extract gt text
                gt_text = segment["text"]

                # Retrieve audio chunk
                chunk = waveform[start_frame: end_frame]

                # TRANSCRIPTION
                beam_results = perform_transcription(
                    model = model, 
                    processor = processor, 
                    audio_chunk = chunk, 
                    # print_beam_results=True,
                    num_beams=3
                )

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
                    "beam_results": beam_results,
                    "start": int(segment["start"]),
                    "end": int(segment["end"]),
                    "gt_text": gt_text
                })

            # ┌───────────────────────────────────────────────┐
            # │                FORMAT & SAVE                  │
            # └───────────────────────────────────────────────┘
            # Format and save a transcript using beam 1's results
            diarized_lines = []
            for data in speaker_separated_data:
                beam_1_text = data["beam_results"]['beam_1']["text"]
                diarized_lines.append(f"({data['start']} - {data['end']})[{data['speaker']}]: {beam_1_text.strip()}\n")

            transcript_file_path = transcripts_folder / "custom_transcript_gt_segments.txt"
            with open(transcript_file_path, "w", encoding="utf-8") as f:
                f.write("".join(diarized_lines))
                
            LOGGER.info(f"Saved transcript for {transcript_file_path}\n\n")      


            # Format and save a JSON using all beams results
            json_results = []
            for data in speaker_separated_data:
                # Start with Ground Truth Text
                result = {
                    "gt": data["gt_text"]
                }

                # Add all Beam results
                result.update(data["beam_results"])

                # Accumulate in a list
                json_results.append(result)

            artifacts_folder = meeting_folder / "artifacts"
            os.makedirs(artifacts_folder, exist_ok = True)
            beam_results_path = artifacts_folder / "beam_results.json"
            with open(beam_results_path, "w") as f:
                json.dump(json_results, f, indent=4)

            LOGGER.info(f"Saved beam results for {beam_results_path}\n\n")  


