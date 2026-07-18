"""Deliverable D: play a canned trajectory and record it — proves the backend.

Usage:
    python scripts/dance_backend_demo.py                 # silent MP4
    python scripts/dance_backend_demo.py --audio song.mp3
    python scripts/dance_backend_demo.py --pip dance.mp4 --audio song.mp3
"""
import argparse
import os
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.camera import SceneCamera
from kuka_sim.dance.trajectory import DanceTrajectory, smooth_and_limit, map_to_workspace
from kuka_sim.dance.player import play
from kuka_sim.dance.recorder import record

USD = "assets/usd/iiwa7_r800.usd"
OUT_DIR = "out"
CAPTURE_EVERY = 4          # 120 Hz sim / 4 -> 30 fps


def canned_trajectory(n=900):
    """A gentle Lissajous figure in the dexterous box (stand-in for real motion)."""
    t = np.linspace(0, 2 * np.pi, n)
    norm = np.stack([0.4 * np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    pos = map_to_workspace(norm, center=[0.5, 0.0, 0.7], half_extents=[0.08, 0.16, 0.12])
    pos = smooth_and_limit(pos, 1.0 / 120.0, max_speed=0.6)
    return DanceTrajectory(ee_pos=pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None)
    ap.add_argument("--pip", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    app, sim = launch(headless=True, enable_cameras=True)
    handles = build_scene(sim, USD, with_probe=False, with_surface=False)
    cam = SceneCamera(pos=(1.9, 1.9, 1.3), target=(0.5, 0.0, 0.7))
    sim.reset(); handles["arm"].initialize(); cam.initialize()

    traj = canned_trajectory()
    frames = []

    def on_step(i):
        if i % CAPTURE_EVERY == 0:
            frames.append(cam.capture())

    play(sim, handles, traj, cam=cam, on_step=on_step)

    out = os.path.join(OUT_DIR, "dance_backend_demo.mp4")
    record(frames, out, fps=30, audio_path=args.audio, pip_video_path=args.pip)
    print(f"[OK] wrote {out} ({len(frames)} frames)", flush=True)
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
