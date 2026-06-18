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