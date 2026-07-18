"""Dance trajectory contract + pure-logic helpers (no Isaac)."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Flange aimed forward (tool axis at +X), matching the force demo.
FORWARD_QUAT = np.array([0.7071, 0.0, 0.7071, 0.0])


@dataclass
class DanceTrajectory:
    ee_pos: np.ndarray                       # (N, 3) target flange positions, world
    ee_quat: Optional[np.ndarray] = None     # (N, 4) wxyz; None -> hold FORWARD_QUAT
    dt: float = 1.0 / 120.0
    cam_pos: Optional[np.ndarray] = None     # (N, 3) or None -> fixed hero camera
    cam_target: Optional[np.ndarray] = None  # (N, 3)
    audio_path: Optional[str] = None
    pip_video_path: Optional[str] = None

    def __len__(self):
        return len(self.ee_pos)


def map_to_workspace(norm_pos, center, half_extents):
    """Map normalized positions in [-1, 1] into the arm's workspace box.

    norm_pos (N,3) with each axis in [-1,1] (clipped) -> center +- half_extents.
    """
    norm = np.clip(np.asarray(norm_pos, float), -1.0, 1.0)
    return np.asarray(center, float) + norm * np.asarray(half_extents, float)


def smooth_and_limit(pos, dt, max_speed, smooth_win=5):
    """Low-pass smooth an EE path (edge-padded moving average), then clamp each
    step so EE speed never exceeds max_speed (m/s). Front-ends produce 'desired'
    motion; this makes it physically playable and graceful."""
    pos = np.asarray(pos, float)
    if smooth_win > 1 and len(pos) >= smooth_win:
        pad = smooth_win // 2
        padded = np.pad(pos, ((pad, pad), (0, 0)), mode="edge")
        k = np.ones(smooth_win) / smooth_win
        pos = np.stack(
            [np.convolve(padded[:, i], k, mode="valid") for i in range(pos.shape[1])],
            axis=1,
        )
    max_step = max_speed * dt
    out = pos.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        dist = float(np.linalg.norm(d))
        if dist > max_step and dist > 0.0:
            out[i] = out[i - 1] + d * (max_step / dist)
    return out
