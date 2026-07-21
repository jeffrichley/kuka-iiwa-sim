"""V1: choreograph the arm to a song and render an MP4 (with the song muxed).

Usage:
    python scripts/dance_music_demo.py --song assets/audio/blue_danube.mp3 \
        --style waltz --camera hero --out out/blue_danube.mp4
    python scripts/dance_music_demo.py --song assets/audio/zarathustra.mp3 \
        --style epic --camera cinematic --seconds 20
"""
import argparse
import os
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.camera import SceneCamera
from kuka_sim.dance.trajectory import smooth_and_limit
from kuka_sim.dance.player import play
from kuka_sim.dance.recorder import record
from kuka_sim.dance.music.features import analyze
from kuka_sim.dance.music.choreographer import choreograph

USD = "assets/usd/iiwa7_r800.usd"
OUT_DIR = "out"
DT = 1.0 / 120.0
CAPTURE_EVERY = 4          # 120 Hz sim / 4 -> 30 fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--style", default="waltz")
    ap.add_argument("--camera", default="hero", choices=["hero", "cinematic"])
    ap.add_argument("--seconds", type=float, default=None,
                    help="render only the first N seconds (fast smoke)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[analyze] {args.song}", flush=True)
    track = analyze(args.song, dt=DT)
    traj = choreograph(track, style=args.style, camera=args.camera)

    if args.seconds is not None:
        cut = int(args.seconds / DT)
        traj.ee_pos = traj.ee_pos[:cut]
        traj.ee_quat = traj.ee_quat[:cut]
        if traj.cam_pos is not None:
            traj.cam_pos = traj.cam_pos[:cut]
            traj.cam_target = traj.cam_target[:cut]

    traj.ee_pos = smooth_and_limit(traj.ee_pos, DT, max_speed=0.6)
    traj.audio_path = args.song
    print(f"[choreograph] {len(traj)} samples, tempo={track.tempo:.1f} bpm", flush=True)

    app, sim = launch(headless=True, enable_cameras=True)
    handles = build_scene(sim, USD, with_probe=False, with_surface=False)
    cam = SceneCamera(pos=(1.9, 1.9, 1.3), target=(0.5, 0.0, 0.7))
    sim.reset(); handles["arm"].initialize(); cam.initialize()

    frames = []

    def on_step(i):
        if i % CAPTURE_EVERY == 0:
            frames.append(cam.capture())

    play(sim, handles, traj, cam=cam, on_step=on_step)

    name = os.path.splitext(os.path.basename(args.song))[0]
    out = args.out or os.path.join(OUT_DIR, f"{name}.mp4")
    record(frames, out, fps=30, audio_path=traj.audio_path)
    print(f"[OK] wrote {out} ({len(frames)} frames)", flush=True)
    _shutdown(app)


def _shutdown(app):
    """Force process exit within seconds of the deliverable landing. Isaac's
    app.close() can deadlock on non-daemon threads; give it a bounded window in
    a daemon thread, then os._exit(0) unconditionally so a finished render never
    lingers as a zombie (and the caller always gets a completion signal)."""
    import sys
    import threading
    sys.stdout.flush(); sys.stderr.flush()
    t = threading.Thread(target=app.close, daemon=True)
    t.start()
    t.join(timeout=15.0)   # let graceful cleanup finish; cap it if it deadlocks
    os._exit(0)            # exit immediately once close() returns or the cap hits


if __name__ == "__main__":
    main()
