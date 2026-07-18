"""FeatureTrack -> DanceTrajectory: combine band-modulated EE moves + a wrist
flourish + an optional camera track. Pure logic (no Isaac)."""
import numpy as np
from kuka_sim.dance.trajectory import DanceTrajectory, map_to_workspace, FORWARD_QUAT
from kuka_sim.dance.music import moves

TWO_PI = 2.0 * np.pi

# Per-style parameter presets. Amplitudes in metres / radians; freqs in Hz.
STYLES = {
    "waltz": dict(
        sway_freq=0.5, sway_amp=0.09, bob_freq=1.0, bob_amp=0.05,
        reach_amp=0.07, beat_amp=0.05, beat_decay=0.25,
        roll_amp=0.25, yaw_amp=0.20, wrist_freq=0.5, onset_amp=0.15,
        cam_orbit_freq=0.03, cam_radius=1.9, cam_push=0.5, cam_height=1.3,
    ),
    "epic": dict(
        sway_freq=0.25, sway_amp=0.10, bob_freq=0.5, bob_amp=0.06,
        reach_amp=0.10, beat_amp=0.06, beat_decay=0.15,
        roll_amp=0.30, yaw_amp=0.25, wrist_freq=0.3, onset_amp=0.20,
        cam_orbit_freq=0.02, cam_radius=2.1, cam_push=0.7, cam_height=1.4,
    ),
}


def euler_to_quat(roll, pitch, yaw):
    """Intrinsic XYZ euler -> wxyz unit quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z], float)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], float)


def choreograph(track, style="waltz", camera="hero",
                box_center=(0.5, 0.0, 0.7), box_half_extents=(0.10, 0.18, 0.14)):
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choices: {sorted(STYLES)}")
    p = STYLES[style]
    n, dt = track.n, track.dt
    t = np.arange(n) * dt
    center = np.asarray(box_center, float)
    half = np.asarray(box_half_extents, float)

    # --- position: band-modulated EE offsets summed in metres ---
    off = np.zeros((n, 3), float)
    off += moves.sway(t, p["sway_freq"], p["sway_amp"]) * track.bass[:, None]
    off += moves.bob(t, p["bob_freq"], p["bob_amp"]) * track.mid[:, None]
    off += moves.reach(t, track.rms, p["reach_amp"])
    off += moves.accent(track.beats, p["beat_decay"], p["beat_amp"], axis=2)
    # global swell: quiet passages small, peaks large
    off *= (0.5 + 0.5 * track.rms)[:, None]

    # normalize by half-extents so map_to_workspace clips into the box
    norm = np.divide(off, half, out=np.zeros_like(off), where=half != 0)
    ee_pos = map_to_workspace(norm, center, half)

    # --- orientation: bounded wrist flourish around FORWARD_QUAT ---
    roll = p["roll_amp"] * track.treble * np.sin(TWO_PI * p["wrist_freq"] * t)
    yaw = p["yaw_amp"] * track.treble * np.cos(TWO_PI * p["wrist_freq"] * t)
    onset_kick = moves.accent(track.onsets, p["beat_decay"], p["onset_amp"], axis=0)[:, 0]
    roll = roll + onset_kick
    ee_quat = np.array([
        quat_mul(euler_to_quat(roll[i], 0.0, yaw[i]), FORWARD_QUAT)
        for i in range(n)
    ])

    # --- camera ---
    cam_pos = cam_target = None
    if camera == "cinematic":
        theta = TWO_PI * p["cam_orbit_freq"] * t
        radius = p["cam_radius"] - p["cam_push"] * track.rms
        cam_pos = np.stack([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            np.full(n, p["cam_height"]),
        ], axis=1)
        cam_target = np.tile(center, (n, 1))
    elif camera != "hero":
        raise ValueError(f"unknown camera {camera!r}; choices: hero, cinematic")

    return DanceTrajectory(ee_pos=ee_pos, ee_quat=ee_quat, dt=dt,
                           cam_pos=cam_pos, cam_target=cam_target)
