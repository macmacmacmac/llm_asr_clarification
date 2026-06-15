import torch
import soundfile as sf

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
    speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate = sample_rate)

    # Collect speech chunks until we hit target duration
    speech_chunks = []
    collected_frames = 0
    target_frames = int(target_duration_sec * sample_rate)

    # Keep collecting speech chunks until we hit target frames
    for segment in speech_timestamps:
        start = segment["start"]
        end = segment["end"]
        chunk = waveform[:, start:end]
        speech_chunks.append(chunk)

        collected_frames += (end - start)
        if collected_frames >= target_frames:
            break

    if len(speech_chunks) > 0:
        # Concatenate speech chunks into a block of continuous speech
        combined_speech = torch.cat(speech_chunks, dim = 1)

        # Trim combined speech chunk so its always target_duration_sec long
        combined_speech = combined_speech[:, :target_frames]

        # Embed the combined speech chunk and return the embedding 
        with torch.no_grad():
            emb = classifier.encode_batch(combined_speech)
    
        return emb.squeeze()

    # Corner case when VAD found 0 speech in the entire audio
    else:
        return None