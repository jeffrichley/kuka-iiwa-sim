"""Isaac-free previewer: build a DanceTrajectory and render the arm as a
dots-and-lines stick figure in matplotlib 3D -> mp4, in seconds.

Runs in the isolated .preview_venv (matplotlib + ikpy + librosa; numpy<2) so it
never touches the Isaac venv. Use it to iterate choreography/amplitude/camera
tuning fast, then do the final beauty render in Isaac once.

The dots are the arm's real joint positions: we solve IK (ikpy, from the URDF)
for each end-effector target, then draw the 8 link frames. It approximates what
Isaac shows (Isaac adds impedance lag); it's for judging motion, not physics.

Usage (preview venv):
    .preview_venv/Scripts/python.exe scripts/preview_trajectory.py \
        --mode music --song assets/audio/blue_danube.mp3 --style waltz \
        --seconds 20 --out out/preview_blue_danube.mp4

    .preview_venv/Scripts/python.exe scripts/preview_trajectory.py \
        --mode video --poses out/dance_studio_poses.npz --scorer energy \
        --seconds 20 --out out/preview_video_energy.mp4
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg
from ikpy.chain import Chain

from kuka_sim.dance.trajectory import smooth_and_limit

URDF = "assets/urdf/lbr_iiwa7_r800_description/iiwa7_r800.urdf"
DT = 1.0 / 120.0
BOX_CENTER = np.array([0.5, 0.0, 0.7])
BOX_HALF = np.array([0.12, 0.18, 0.15])


def build_music_traj(song, style, camera):
    from kuka_sim.dance.music.features import analyze
    from kuka_sim.dance.music.choreographer import choreograph
    track = analyze(song, dt=DT)
    return choreograph(track, style=style, camera=camera,
                       box_center=tuple(BOX_CENTER), box_half_extents=tuple(BOX_HALF))


def build_video_traj(poses, scorer, song):
    from kuka_sim.dance.video.pose import load_pose_track
    from kuka_sim.dance.video.scorers import SCORERS, ScorerCtx
    from kuka_sim.dance.video.retarget import retarget
    track = load_pose_track(poses)
    gray = np.load(poses)["gray"]
    ctx = ScorerCtx()
    if scorer in ("music", "blend"):
        import librosa
        y, sr = librosa.load(song, sr=None, mono=True)
        rms = librosa.feature.rms(y=y)[0]
        rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        env = np.interp(np.arange(len(track.xy)) / track.fps, rms_t, rms)
        lo, hi = float(env.min()), float(env.max())
        ctx.beat_env = (env - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros(len(env))
    if scorer == "saliency":
        ctx.gray_frames = gray
    focus = SCORERS[scorer](track, ctx)
    return retarget(focus, track.fps, dt=DT,
                    box_center=tuple(BOX_CENTER), box_half_extents=tuple(BOX_HALF))


def ik_joint_positions(ee_pos):
    """IK each EE target -> the 3D positions of every link frame (dots)."""
    chain = Chain.from_urdf_file(URDF, base_elements=["lbr_link_0"])
    q = np.zeros(len(chain.links))
    dots = []
    for target in ee_pos:
        q = chain.inverse_kinematics(target, initial_position=q)
        fk = chain.forward_kinematics(q, full_kinematics=True)
        dots.append(np.array([T[:3, 3] for T in fk]))
    return np.array(dots)   # (F, n_links, 3)


def _box_edges(center, half):
    c, h = center, half
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = c + corners * h
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs((corners[i] - corners[j]) > 1e-9)) == 1:  # share 2 coords
                edges.append((corners[i], corners[j]))
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["music", "video"], required=True)
    ap.add_argument("--song", default="assets/audio/blue_danube.mp3")
    ap.add_argument("--style", default="waltz")
    ap.add_argument("--camera", default="hero")
    ap.add_argument("--poses", default="out/dance_studio_poses.npz")
    ap.add_argument("--scorer", default="energy")
    ap.add_argument("--dancer", default=None,
                    help="source dance clip to show side-by-side with the arm")
    ap.add_argument("--start", type=float, default=0.0,
                    help="preview from this many seconds into the song/clip")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--max-speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    if args.mode == "music":
        traj = build_music_traj(args.song, args.style, args.camera)
    else:
        traj = build_video_traj(args.poses, args.scorer, args.song)

    ee = traj.ee_pos
    start_i = int(args.start / DT)
    ee = ee[start_i:]
    if args.seconds is not None:
        ee = ee[:int(args.seconds / DT)]
    ee = smooth_and_limit(ee, DT, max_speed=args.max_speed)

    step = max(1, int(round(1.0 / DT / args.fps)))   # 120Hz -> fps
    ee_s = ee[::step]
    travel = (ee_s.max(0) - ee_s.min(0)) * 100
    print(f"[preview] {len(ee_s)} frames; EE travel cm x={travel[0]:.1f} "
          f"y={travel[1]:.1f} z={travel[2]:.1f}", flush=True)

    dots = ik_joint_positions(ee_s)                  # (F, L, 3)

    fig = plt.figure(figsize=(6, 6), dpi=100)   # -> 600x600, known for side-by-side
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.2, 0.8); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 1.0)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=-60)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    for a, b in _box_edges(BOX_CENTER, BOX_HALF):
        ax.plot(*zip(a, b), color="tab:orange", lw=0.6, alpha=0.5)

    link_line, = ax.plot([], [], [], "-o", color="tab:blue", lw=3, ms=5, mfc="white")
    ee_dot, = ax.plot([], [], [], "o", color="crimson", ms=8)
    trail, = ax.plot([], [], [], "-", color="crimson", lw=1, alpha=0.5)
    title = ax.set_title("")

    def update(i):
        d = dots[i]
        link_line.set_data(d[:, 0], d[:, 1]); link_line.set_3d_properties(d[:, 2])
        ee_dot.set_data([d[-1, 0]], [d[-1, 1]]); ee_dot.set_3d_properties([d[-1, 2]])
        lo = max(0, i - 30)
        tp = dots[lo:i + 1, -1]
        trail.set_data(tp[:, 0], tp[:, 1]); trail.set_3d_properties(tp[:, 2])
        title.set_text(f"{args.mode}:{args.style if args.mode=='music' else args.scorer}"
                       f"  frame {i}/{len(dots)}")
        return link_line, ee_dot, trail, title

    ani = FuncAnimation(fig, update, frames=len(dots), interval=1000 / args.fps, blit=False)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams["animation.ffmpeg_path"] = ff
    silent = args.out + ".silent.mp4"
    ani.save(silent, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    _finish(ff, silent, args.song, args.out, args.start, args.fps, args.dancer)
    print(f"[OK] wrote {args.out} ({len(dots)} frames)", flush=True)


def _finish(ffmpeg, silent_path, song, out_path, start, fps, dancer=None, arm_h=600):
    """Produce the final preview: mux the song, and — if a dancer clip is given —
    show it side-by-side (dancer left, arm right, time-aligned) so you can see
    whether the arm follows her. Falls back to the silent file if ffmpeg fails."""
    import subprocess
    if dancer is not None:
        # dancer scaled to the arm's height, re-timed to the arm's fps, placed
        # left of the arm; song muxed from `start`, trimmed to the shorter stream.
        filt = (f"[1:v]fps={fps},scale=-2:{arm_h}[d];[d][0:v]hstack=inputs=2[v]")
        cmd = [ffmpeg, "-y", "-i", silent_path, "-ss", str(start), "-i", dancer,
               "-ss", str(start), "-i", song, "-filter_complex", filt,
               "-map", "[v]", "-map", "2:a", "-c:a", "aac", "-shortest", out_path]
    else:
        cmd = [ffmpeg, "-y", "-i", silent_path, "-ss", str(start), "-i", song,
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
               "-shortest", out_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(silent_path)
    except subprocess.CalledProcessError as e:
        print("[warn] ffmpeg finish failed; keeping silent preview:\n"
              + e.stderr.decode(errors="replace")[-600:], flush=True)
        os.replace(silent_path, out_path)


if __name__ == "__main__":
    main()
