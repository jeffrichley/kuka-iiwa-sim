"""Retarget a human arm onto the iiwa in FULL 3D: make the robot's three arm
segments (upper arm, forearm, hand) point along HER segments' actual 3D
directions -- depth included, not just the frontal plane.

Rather than hand-derive inverse kinematics for the real link geometry, we solve
against the actual robot forward kinematics: for each frame, find the 7 joint
angles that make the robot's segment directions match hers (least-squares,
warm-started from the previous frame, bounded by the joint limits).

Input: MediaPipe world landmarks (F, 33, 3), metres, hip origin.
World frame (verified): +x = her right, +y = DOWN, +z = toward camera.
Output: (F, 7) iiwa joint angles A1..A7 (radians), within limits.
"""
import os
import numpy as np

R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16
L_SHOULDER, L_ELBOW, L_WRIST = 11, 13, 15

IIWA_LIMITS = np.deg2rad([170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175.0])

# chain FK dots (0 base, 1..7 A1..A7, 8 flange); arm segments between these:
_SEG = [(2, 4), (4, 6), (6, 8)]          # upper arm, forearm, hand
_URDF = os.path.join("assets", "urdf", "lbr_iiwa7_r800_description", "iiwa7_r800.urdf")
_CHAIN = None


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _arm_indices(side):
    if side == "right":
        return R_SHOULDER, R_ELBOW, R_WRIST, (18, 20)   # +pinky, index
    if side == "left":
        return L_SHOULDER, L_ELBOW, L_WRIST, (17, 19)
    raise ValueError(f"side must be 'right' or 'left', got {side!r}")


def _to_robot(v):
    """Her world dir (x right, y DOWN, z toward cam) -> robot dir (x fwd, y left,
    z up): up = -y, forward = +z (toward camera), left = -x."""
    return np.array([v[2], -v[0], -v[1]])


def _chain():
    global _CHAIN
    if _CHAIN is None:
        from ikpy.chain import Chain
        _CHAIN = Chain.from_urdf_file(_URDF, base_elements=["lbr_link_0"])
    return _CHAIN


def _seg_dirs(q7, chain):
    fk = chain.forward_kinematics(np.concatenate([[0.0], q7, [0.0]]),
                                  full_kinematics=True)
    d = np.array([T[:3, 3] for T in fk])
    return [_unit(d[b] - d[a]) for a, b in _SEG]


def mimic_joint_traj(xyz, side="right", gains=None):
    """(F,33,3) world landmarks -> (F,7) iiwa joint angles matching her arm in 3D."""
    from scipy.optimize import least_squares
    sh_i, el_i, wr_i, hand_i = _arm_indices(side)
    chain = _chain()
    xyz = np.asarray(xyz, float)
    F = len(xyz)
    q = np.zeros((F, 7))
    q_prev = np.zeros(7)
    for i in range(F):
        p = xyz[i]
        hand_pt = (p[hand_i[0]] + p[hand_i[1]]) / 2.0
        targets = [_to_robot(_unit(p[el_i] - p[sh_i])),
                   _to_robot(_unit(p[wr_i] - p[el_i])),
                   _to_robot(_unit(hand_pt - p[wr_i]))]

        def resid(qq):
            dirs = _seg_dirs(qq, chain)
            return np.concatenate([dirs[j] - targets[j] for j in range(3)])

        sol = least_squares(resid, q_prev, bounds=(-IIWA_LIMITS, IIWA_LIMITS),
                            method="trf", max_nfev=60, xtol=1e-3, ftol=1e-3)
        q[i] = sol.x
        q_prev = sol.x
    return q
