"""Focus path (normalized image coords) -> DanceTrajectory on the sim grid."""
import numpy as np
from kuka_sim.dance.trajectory import DanceTrajectory, map_to_workspace


def retarget(focus_xy, fps, dt=1.0 / 120.0,
             box_center=(0.5, 0.0, 0.7), box_half_extents=(0.10, 0.18, 0.14),
             depth_env=None, video_path=None, audio_path=None):
    """Map a (F,2) normalized image-space focus path to an EE trajectory.

    image-x [0,1] -> box lateral (y); (1 - image-y) -> box height (z); depth (x)
    from depth_env (in [-1,1]) or held at the box center. Resample F@fps to the
    sim grid, then map into the workspace box. pip/audio point at the source clip.
    """
    focus_xy = np.asarray(focus_xy, float)
    F = len(focus_xy)
    center = np.asarray(box_center, float)
    half = np.asarray(box_half_extents, float)

    norm = np.zeros((F, 3), float)
    norm[:, 1] = 2.0 * focus_xy[:, 0] - 1.0           # x right -> lateral y
    norm[:, 2] = 2.0 * (1.0 - focus_xy[:, 1]) - 1.0   # y down  -> height z (flip)
    if depth_env is not None:
        norm[:, 0] = np.asarray(depth_env, float)

    # resample F@fps -> N@dt
    n = int(round(F / fps / dt))
    src_t = np.arange(F) / fps
    dst_t = np.arange(n) * dt
    norm_rs = np.stack([np.interp(dst_t, src_t, norm[:, c]) for c in range(3)], axis=1)

    ee_pos = map_to_workspace(norm_rs, center, half)
    return DanceTrajectory(
        ee_pos=ee_pos, dt=dt,
        pip_video_path=video_path,
        audio_path=audio_path if audio_path is not None else video_path,
    )
