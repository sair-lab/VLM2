import os
import re
from typing import List, Tuple, Dict, Any, Optional

def get_frame_number_from_filename(filename: str) -> Optional[int]:
    basename = os.path.splitext(filename)[0]
    numbers = re.findall(r'\d+', basename)
    if not numbers:
        return None
    
    try:
        return int(numbers[-1])
    except ValueError:
        return None

def calculate_frame_timestamps(filenames: List[str], fps: float) -> Tuple[List[Dict[str, Any]], float]:
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    if not filenames:
        return [], 0.0

    extracted_frames: List[Dict[str, Any]] = []
    for fname in filenames:
        frame_num = get_frame_number_from_filename(fname)
        if frame_num is not None:
            extracted_frames.append({'original_filename': fname, 'frame_number': frame_num})
        else:
            print(f"Warning: Could not extract a frame number from filename '{fname}'. Skipping this file.")

    if not extracted_frames:
        return [], 0.0

    sorted_frames = sorted(extracted_frames, key=lambda x: x['frame_number'])
    num_sampled_frames = len(sorted_frames)

    if num_sampled_frames == 1:
        sorted_frames[0]['timestamp'] = 0.0
        return sorted_frames, 0.0

    n_first = sorted_frames[0]['frame_number']
    n_last = sorted_frames[-1]['frame_number']
    duration_orig_span = (n_last - n_first) / fps

    if duration_orig_span == 0:
        delta_t_sample = 0.0
    else:
        delta_t_sample = duration_orig_span / (num_sampled_frames - 1)

    for i, frame_info in enumerate(sorted_frames):
        frame_info['timestamp'] = i * delta_t_sample
    total_sequence_time = duration_orig_span
    
    if num_sampled_frames > 1:
        total_sequence_time = (num_sampled_frames - 1) * delta_t_sample

    return sorted_frames, total_sequence_time