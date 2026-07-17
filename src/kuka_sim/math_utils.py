"""Quaternion helpers (wxyz convention). Pure NumPy — no Isaac dependency."""
import numpy as np


def quat_conj(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    aw, ax, ay, az = np.asarray(a, dtype=float)
    bw, bx, by, bz = np.asarray(b, dtype=float)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def orientation_error(q_des, q_cur):
    """Axis-angle (3,) rotation taking q_cur -> q_des, in the current frame."""
    q_err = quat_mul(np.asarray(q_des, float), quat_conj(q_cur))
    if q_err[0] < 0.0:                    # shortest path
        q_err = -q_err
    vec = q_err[1:]
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(norm, q_err[0])
    return vec / norm * angle
