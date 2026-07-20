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
    # Anatomical: the 3 bend joints map to shoulder / elbow / wrist, each segment
    # pointing where hers points (frontal-plane angle). The torso's twist & side-
    # lean take the roll joints; its forward-lean merges into the shoulder.
    a1_twist=1.2,     # A1 roll  <- torso twist
    a2_gain=1.0,      # A2 bend  <- shoulder: upper-arm angle from vertical
    a3_bend=1.6,      # A3 roll  <- torso side-lean
    a4_gain=-1.0,     # A4 bend  <- elbow: forearm angle change (neg: -y axis)
    a6_gain=-1.0,     # A6 bend  <- wrist: hand angle change
)


def _frontal_world(v):
    """Angle of v in the image/frontal plane: 0 = up, +pi/2 = image-right, +-pi = down."""
    return float(np.arctan2(v @ X_AXIS, v @ WORLD_UP))


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
    hand_i = (18, 20) if side == "right" else (17, 19)   # pinky, index (fingertips)
    twist = np.zeros(F); side_bend = np.zeros(F)
    upper_ang = np.zeros(F); fore_ang = np.zeros(F); hand_ang = np.zeros(F)
    for i in range(F):
        p = xyz[i]
        up, across, facing = _torso_frame(p)
        side_bend[i] = up @ X_AXIS                          # spine tilts along image-x
        twist[i] = np.arcsin(np.clip(across @ Z_AXIS, -1.0, 1.0))  # torso twist (~0 frontal)
        # each segment's angle in the image/frontal plane (world-referenced, so
        # the shoulder/elbow/wrist bends match what the viewer sees)
        sh, el, wr = p[sh_i], p[el_i], p[wr_i]
        hand_pt = (p[hand_i[0]] + p[hand_i[1]]) / 2.0
        upper_ang[i] = _frontal_world(el - sh)              # upper arm
        fore_ang[i] = _frontal_world(wr - el)               # forearm
        hand_ang[i] = _frontal_world(hand_pt - wr)          # hand

    # Unwrap over TIME so a segment sweeping through +-pi doesn't snap 360deg.
    upper_ang = np.unwrap(upper_ang)
    fore_ang = np.unwrap(fore_ang)
    hand_ang = np.unwrap(hand_ang)

    q = np.zeros((F, 7))
    q[:, 0] = g["a1_twist"] * twist                    # A1 <- torso twist
    q[:, 1] = g["a2_gain"] * upper_ang                 # A2 shoulder <- upper-arm angle
    q[:, 2] = g["a3_bend"] * side_bend                 # A3 <- torso side-lean
    q[:, 3] = g["a4_gain"] * (fore_ang - upper_ang)    # A4 elbow  <- forearm turns off upper arm
    q[:, 5] = g["a6_gain"] * (hand_ang - fore_ang)     # A6 wrist  <- hand turns off forearm
    return np.clip(q, -IIWA_LIMITS, IIWA_LIMITS)
