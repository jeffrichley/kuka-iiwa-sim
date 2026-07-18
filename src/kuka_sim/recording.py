"""Encode a sequence of RGB frames to an MP4."""
import numpy as np


def frames_to_mp4(frames, path, fps=30):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (255.0 * np.clip(arr, 0.0, 1.0)).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[-1] == 4:   # drop alpha
                arr = arr[..., :3]
            writer.append_data(arr)
    return path
