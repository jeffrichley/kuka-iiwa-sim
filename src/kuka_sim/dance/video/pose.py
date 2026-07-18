"""Dance video -> MediaPipe pose keypoints (PoseTrack). Seams are mocked in tests."""
from dataclasses import dataclass
import numpy as np

N_LANDMARKS = 33


@dataclass
class PoseTrack:
    xy: np.ndarray        # (F, K, 2) normalized image coords [0,1], x right / y down
    visible: np.ndarray   # (F, K) visibility in [0,1]
    fps: float


def _fill_gaps(xy):
    """Linear-interpolate NaNs per landmark/coord over frames; all-NaN -> 0.5."""
    xy = np.asarray(xy, float).copy()
    F, K, C = xy.shape
    idx = np.arange(F)
    for k in range(K):
        for c in range(C):
            col = xy[:, k, c]
            valid = ~np.isnan(col)
            if valid.sum() == 0:
                xy[:, k, c] = 0.5
            elif valid.sum() < F:
                xy[:, k, c] = np.interp(idx, idx[valid], col[valid])
    return xy


def _read_frames(video_path, max_frames=None):
    """Read RGB frames + fps via imageio. Seam: monkeypatched in tests."""
    import imageio.v3 as iio
    try:
        fps = float(iio.immeta(video_path, plugin="pyav").get("fps", 30.0))
    except Exception:
        fps = 30.0
    frames = []
    for i, fr in enumerate(iio.imiter(video_path)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(np.asarray(fr)[:, :, :3])
    return frames, fps


def _detect_landmarks(frames):
    """Run MediaPipe Pose per frame -> (xy (F,K,2) with NaN gaps, visible (F,K))."""
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    F = len(frames)
    xy = np.full((F, N_LANDMARKS, 2), np.nan)
    vis = np.zeros((F, N_LANDMARKS))
    for i, fr in enumerate(frames):
        res = pose.process(np.ascontiguousarray(fr))
        if res.pose_landmarks:
            for k, lm in enumerate(res.pose_landmarks.landmark):
                xy[i, k] = [lm.x, lm.y]
                vis[i, k] = lm.visibility
    pose.close()
    return xy, vis


def estimate_poses(video_path, max_frames=None):
    frames, fps = _read_frames(video_path, max_frames)
    xy, vis = _detect_landmarks(frames)
    return PoseTrack(xy=_fill_gaps(xy), visible=np.asarray(vis, float), fps=fps)
