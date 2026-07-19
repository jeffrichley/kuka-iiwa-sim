"""Retarget a human upper body onto the iiwa's joints so the robot *mimics* her:
her torso lean/bend/twist drives the base joints, her arm (measured RELATIVE to
the torso) drives the upper joints, and her elbow/wrist drive the tip. This
"unrolls" her waist->hand chain onto the robot's base->tool chain.

Input: MediaPipe world landmarks (F, 33, 3), metres, hip origin.
World frame (verified): +x = her right, +y = DOWN, +z = toward camera.
Output: (F, 7) iiwa joint angles A1..A7 (radians), clamped to limits.

Signs/gains are tuned pose-by-pose against the still renderer; every value here
is a knob in DEFAULT_GAINS.
"""
import numpy as np

# MediaPipe Pose landmark indices
R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16
L_SHOULDER, L_ELBOW, L_WRIST = 11, 13, 15
R_HIP, L_HIP = 24, 23

IIWA_LIMITS = np.deg2rad([170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175.0])

X_AXIS = np.array([1.0, 0.0, 0.0])    # image left/right
Z_AXIS = np.array([0.0, 0.0, 1.0])    # toward camera
WORLD_UP = np.array([0.0, -1.0, 0.0])  # +y is down, so up is -y

DEFAULT_GAINS = dict(
    # "Unroll the chain": torso -> base joints, arm -> upper joints (no sharing).
    # --- torso -> base (A1 yaw, A2 fold, A3 side-lean) ---
    a1_twist=1.2,     # shoulders rotate over hips -> base yaw
    a2_fold=1.4,      # torso bows forward/back -> base folds (THE lean you wanted)
    a3_bend=1.6,      # torso side-bend -> roll
    # --- arm, relative to the torso -> upper joints (A4 shoulder, A6 elbow) ---
    a4_gain=1.3,      # arm lifts off the spine -> A4 (shoulder elevation)
    a6_gain=-1.4,     # elbow flexion -> A6 (elbow)
)


def _frontal(v, across, up):
    """Signed angle of v in the torso frontal plane: 0 = up the spine,
    +pi/2 = out to her right (across), -pi/2 = her left, +-pi = down."""
    return float(np.arctan2(v @ across, v @ up))


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


def _torso_frame(p):
    """Orthonormal torso frame from the 4-point box: (up, across, facing)."""
    sh_c = (p[L_SHOULDER] + p[R_SHOULDER]) / 2.0
    hip_c = (p[L_HIP] + p[R_HIP]) / 2.0
    up = _unit(sh_c - hip_c)                    # spine, points toward head
    across = _unit(p[R_SHOULDER] - p[L_SHOULDER])   # shoulder line, toward her right
    facing = _unit(np.cross(up, across))        # box normal, toward camera (neutral)
    across = _unit(np.cross(facing, up))        # re-orthogonalize
    return up, across, facing


def mimic_joint_traj(xyz, side="right", gains=None):
    """(F,33,3) world landmarks -> (F,7) iiwa joint angles (torso-rooted chain)."""
    g = {**DEFAULT_GAINS, **(gains or {})}
    sh_i, el_i, wr_i = _arm_indices(side)
    xyz = np.asarray(xyz, float)
    F = len(xyz)
    q = np.zeros((F, 7))
    for i in range(F):
        p = xyz[i]
        up, across, facing = _torso_frame(p)

        # --- torso signals (relative to the world) ---
        lean_fwd = float(up @ Z_AXIS)                       # bow toward camera +, arch -
        side_bend = float(up @ X_AXIS)                      # spine tilts along image-x +
        # twist = how far the shoulder line rotates OUT of the image plane (depth).
        # ~0 when frontal (shoulders across the image), grows as she turns.
        twist = float(np.arcsin(np.clip(across @ Z_AXIS, -1.0, 1.0)))

        # --- arm as SIGNED frontal-plane angle RELATIVE TO THE TORSO, so it
        # composes with the base lean (the base folds like her torso, the arm
        # rides on top). 0 = along the spine, +pi/2 = out across, +-pi = down. ---
        upper = p[el_i] - p[sh_i]
        fore = p[wr_i] - p[el_i]
        upper_ang = _frontal(upper, across, up)
        fore_ang = _frontal(fore, across, up)
        flex = _angle(upper, fore)                          # elbow: 0 straight .. pi folded

        # torso forward/back fold (spine tilt in the sagittal/depth plane)
        sagittal = float(np.arctan2(up @ Z_AXIS, up @ WORLD_UP))
        q[i, 0] = g["a1_twist"] * twist       # base yaw   <- torso twist
        q[i, 1] = g["a2_fold"] * sagittal     # base fold  <- torso bow (the lean)
        q[i, 2] = g["a3_bend"] * side_bend    # base roll  <- torso side-lean
        q[i, 3] = g["a4_gain"] * upper_ang    # shoulder   <- arm lift rel. torso
        q[i, 5] = g["a6_gain"] * flex         # elbow      <- her elbow flexion

    return np.clip(q, -IIWA_LIMITS, IIWA_LIMITS)
