"""
models/video_utils.py -- tiny shared helpers for sampling a fixed number of frames from a
video file, used by every "path is a video file" model wrapper (models/trn.py,
models/trn_official.py, models/r3d_ucf101.py) so the segment-center sampling logic exists
in exactly one place.
"""
import numpy as np
from torchcodec.decoders import VideoDecoder


def sample_segment_centers(n_total: int, n_samples: int) -> np.ndarray:
    """Uniformly divide a video of n_total frames into n_samples segments and return each
    segment's centre frame index (the TSN/TRN "test_mode" sampling protocol)."""
    seg = n_total / n_samples
    idx = np.floor(np.arange(n_samples) * seg + seg / 2).astype(np.int64)
    return np.clip(idx, 0, n_total - 1)


def sample_frames(video_path, n_frames: int):
    """video_path -> (n_frames, C, H, W) uint8 RGB tensor, centre frame of each of n_frames
    uniformly-sized segments."""
    decoder = VideoDecoder(str(video_path))
    idx = sample_segment_centers(len(decoder), n_frames)
    return decoder.get_frames_at(indices=idx.tolist()).data
