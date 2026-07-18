"""V2: drive the arm from a dance video (pose -> saliency scorer -> retarget) and
render it with the source clip inset as picture-in-picture.

Two stages (pose lives in an isolated venv so mediapipe/opencv can't clobber
Isaac's cv2 — see scripts/extract_pose.py):

  1) .pose_venv/Scripts/python.exe scripts/extract_pose.py \
         --video assets/video/dance.mp4 --out out/dance_poses.npz
  2) env_isaaclab/Scripts/python.exe scripts/dance_video_demo.py \
         --video assets/video/dance.mp4 --poses out/dance_poses.npz \
         --scorer blend --out out/dance_video.mp4

The .npz carries keypoints + downsampled gray frames, so this stage needs no
video decoding — only the song audio (for music/blend) is read here via librosa.
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
from kuka_sim.dance.video.pose import load_pose_track
from kuka_sim.dance.video.scorers import SCORERS, ScorerCtx
from kuka_sim.dance.video.retarget import retarget

USD = "assets/usd/iiwa7_r800.usd"
OUT_DIR = "out"
DT = 1.0 / 120.0
CAPTURE_EVERY = 4


def _beat_env(video_path, n_frames, fps):
    """Per-frame music energy from the video's audio (librosa), on the frame grid.
    Best-effort: returns zeros if the audio can't be decoded."""
    import librosa
    try:
        y, sr = librosa.load(video_path, sr=None, mono=True)
    except Exception as e:
        print(f"[warn] librosa could not read audio ({e}); beat_env=0", flush=True)
        return np.zeros(n_frames)
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    frame_t = np.arange(n_frames) / fps
    env = np.interp(frame_t, rms_t, rms)
    lo, hi = float(env.min()), float(env.max())
    return (env - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros(n_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="source clip (for PiP + audio)")
    ap.add_argument("--poses", required=True, help=".npz from scripts/extract_pose.py")
    ap.add_argument("--scorer", default="blend", choices=list(SCORERS))
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--no-pip", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[poses] {args.poses}", flush=True)
    track = load_pose_track(args.poses)
    gray = np.load(args.poses)["gray"]
    if args.seconds is not None:
        cut = int(args.seconds * track.fps)
        track.xy = track.xy[:cut]; track.visible = track.visible[:cut]; gray = gray[:cut]
    F = len(track.xy)

    ctx = ScorerCtx()
    if args.scorer in ("music", "blend"):
        ctx.beat_env = _beat_env(args.video, F, track.fps)
    if args.scorer == "saliency":
        ctx.gray_frames = gray

    focus = SCORERS[args.scorer](track, ctx)
    pip = None if args.no_pip else args.video
    traj = retarget(focus, track.fps, dt=DT, video_path=pip, audio_path=args.video)
    traj.ee_pos = smooth_and_limit(traj.ee_pos, DT, max_speed=0.6)
    print(f"[retarget] scorer={args.scorer} {len(traj)} samples", flush=True)

    app, sim = launch(headless=True, enable_cameras=True)
    handles = build_scene(sim, USD, with_probe=False, with_surface=False)
    cam = SceneCamera(pos=(1.9, 1.9, 1.3), target=(0.5, 0.0, 0.7))
    sim.reset(); handles["arm"].initialize(); cam.initialize()

    frames_out = []

    def on_step(i):
        if i % CAPTURE_EVERY == 0:
            frames_out.append(cam.capture())

    play(sim, handles, traj, cam=cam, on_step=on_step)

    name = os.path.splitext(os.path.basename(args.video))[0]
    out = args.out or os.path.join(OUT_DIR, f"{name}_{args.scorer}.mp4")
    record(frames_out, out, fps=30, audio_path=traj.audio_path,
           pip_video_path=traj.pip_video_path)
    print(f"[OK] wrote {out} ({len(frames_out)} frames)", flush=True)
    _shutdown(app)


def _shutdown(app):
    """Force process exit within seconds of the deliverable landing (Isaac's
    app.close() can deadlock before os._exit). See dance_music_demo._shutdown."""
    import sys
    import threading
    sys.stdout.flush(); sys.stderr.flush()
    t = threading.Thread(target=app.close, daemon=True)
    t.start()
    t.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
