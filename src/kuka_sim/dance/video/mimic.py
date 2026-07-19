"""Retarget a human arm's 3D pose onto the iiwa's joint angles so the robot
*mimics* her arm (elbow bends when hers bends), instead of chasing a point.

Input: MediaPipe world landmarks, (F, 33, 3) in metres, origin at the hips.
World frame (verified on the studio clip): +x right, **+y down**, +z toward camera.
Output: (F, 7) iiwa joint angles A1..A7 (radians).

The mapping is deliberately simple and legible (tune the gains against the
previewer):
  A1 base yaw     <- her upper-arm azimuth  (arm swings across -> base rotates)
  A2 shoulder     <- her shoulder elevation (arm lifts -> robot shoulder lifts)
  A4 elbow        <- her elbow flexion      (her elbow bends -> robot elbow bends)
  A6 wrist pitch  <- her forearm elevation  (hand up/down -> wrist follows)
A3/A5/A7 (rolls) are hard to read from 3-point limbs, so they rest at 0.
"""
import numpy as np

# MediaPipe Pose landmark indices
R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16
L_SHOULDER, L_ELBOW, L_WRIST = 11, 13, 15
R_HIP, L_HIP = 24, 23

# iiwa 7 R800 joint limits (rad), A1..A7.
IIWA_LIMITS = np.deg2rad([170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175.0])

DOWN = np.array([0.0, 1.0, 0.0])   # world +y points down

DEFAULT_GAINS = dict(
    a1_gain=1.2, a2_gain=1.3, a4_gain=1.4, a6_gain=1.0,
    a2_offset=-np.pi / 2,   # center A2 so a horizontal arm -> ~0
    a4_offset=0.0,
)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _angle(a, b):
    return float(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0)))


def _arm_indices(side):
    if side == "right":
        return R_SHOULDER, R_ELBOW, R_WRIST
    if side == "left":
        return L_SHOULDER, L_ELBOW, L_WRIST
    raise ValueError(f"side must be 'right' or 'left', got {side!r}")


def mimic_joint_traj(xyz, side="right", gains=None):
    """(F,33,3) world landmarks -> (F,7) iiwa joint angles mimicking that arm."""
    g = {**DEFAULT_GAINS, **(gains or {})}
    sh_i, el_i, wr_i = _arm_indices(side)
    xyz = np.asarray(xyz, float)
    F = len(xyz)
    q = np.zeros((F, 7))
    for i in range(F):
        upper = xyz[i, el_i] - xyz[i, sh_i]    # shoulder -> elbow
        fore = xyz[i, wr_i] - xyz[i, el_i]     # elbow -> wrist

        # shoulder elevation: 0 = arm hanging down, pi = arm straight up
        elev = _angle(upper, DOWN)
        # azimuth of the upper arm in the horizontal (x-z) plane
        azim = np.arctan2(upper[2], upper[0])
        # elbow flexion: 0 = straight, larger = more bent
        flex = _angle(upper, fore)
        # forearm elevation (drives the wrist so the hand's up/down reads)
        fore_elev = _angle(fore, DOWN)

        q[i, 0] = g["a1_gain"] * azim                       # A1 base yaw
        q[i, 1] = g["a2_gain"] * (elev + g["a2_offset"])    # A2 shoulder
        q[i, 3] = -g["a4_gain"] * (flex + g["a4_offset"])   # A4 elbow
        q[i, 5] = g["a6_gain"] * (fore_elev - np.pi / 2)    # A6 wrist pitch

    return np.clip(q, -IIWA_LIMITS, IIWA_LIMITS)
