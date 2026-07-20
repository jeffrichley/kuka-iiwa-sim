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

# colored chain segments (must match render_pose_still): torso/upper/fore/hand,
# ~2 robot links each (chain dots 0 base..8 flange).
ROBOT_SEGMENTS = [(0, 2, "#d62728"), (2, 4, "#1f77b4"), (4, 6, "#2ca02c"), (6, 8, "#ff7f0e")]


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


def fk_joint_positions(q7):
    """(F,7) iiwa joint angles -> link positions (F,L,3) + flange transforms (F,4,4).
    The chain is [base(fixed), A1..A7, ee(fixed)], so pad each side with 0."""
    chain = Chain.from_urdf_file(URDF, base_elements=["lbr_link_0"])
    dots, flanges = [], []
    for q in q7:
        q_full = np.concatenate([[0.0], q, [0.0]])          # base + 7 + ee
        fk = chain.forward_kinematics(q_full, full_kinematics=True)
        dots.append(np.array([T[:3, 3] for T in fk]))
        flanges.append(fk[-1])
    return np.array(dots), np.array(flanges)


def _hand_lines(flange_T):
    """4 finger segments (along approach) + 1 thumb (palm axis) at the flange."""
    ee = flange_T[:3, 3]; approach = flange_T[:3, 2]; palm_x = flange_T[:3, 0]
    segs = []
    for s in (-1.5, -0.5, 0.5, 1.5):
        base = ee + palm_x * (s * 0.02)
        segs.append((base, base + approach * 0.09))
    segs.append((ee, ee + palm_x * 0.06 + approach * 0.03))   # thumb (last)
    return segs


def _smooth_q(q, win):
    """Edge-padded moving average over time for each joint (de-jitter the pose)."""
    if win <= 1 or len(q) < win:
        return q
    pad_l, pad_r = win // 2, win - 1 - win // 2
    qp = np.pad(q, ((pad_l, pad_r), (0, 0)), mode="edge")
    k = np.ones(win) / win
    return np.stack([np.convolve(qp[:, j], k, mode="valid") for j in range(q.shape[1])], axis=1)


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
    ap.add_argument("--mode", choices=["music", "video", "mimic"], required=True)
    ap.add_argument("--song", default="assets/audio/blue_danube.mp3")
    ap.add_argument("--style", default="waltz")
    ap.add_argument("--camera", default="hero")
    ap.add_argument("--poses", default="out/dance_studio_poses.npz")
    ap.add_argument("--scorer", default="energy")
    ap.add_argument("--side", default="right", choices=["right", "left"])
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

    if args.mode == "mimic":
        # joint-space: her arm's 3D angles -> robot joints -> FK. Runs at the
        # video's own fps so it's frame-aligned with the dancer side-by-side.
        from kuka_sim.dance.video.mimic import mimic_joint_traj
        d = np.load(args.poses)
        xyz = np.nan_to_num(d["xyz"].astype(float), nan=0.0)
        out_fps = float(d["fps"])
        s0 = int(args.start * out_fps)
        xyz = xyz[s0:]
        if args.seconds is not None:
            xyz = xyz[:int(args.seconds * out_fps)]
        q = mimic_joint_traj(xyz, side=args.side)
        q = _smooth_q(q, max(1, int(round(out_fps * 0.15))))     # ~150 ms
        dots, flanges = fk_joint_positions(q)
        label = f"mimic:{args.side}"
    else:
        if args.mode == "music":
            traj = build_music_traj(args.song, args.style, args.camera)
            label = args.style
        else:
            traj = build_video_traj(args.poses, args.scorer, args.song)
            label = args.scorer
        ee = traj.ee_pos[int(args.start / DT):]
        if args.seconds is not None:
            ee = ee[:int(args.seconds / DT)]
        ee = smooth_and_limit(ee, DT, max_speed=args.max_speed)
        step = max(1, int(round(1.0 / DT / args.fps)))   # 120Hz -> fps
        dots = ik_joint_positions(ee[::step])            # (F, L, 3)
        flanges = None
        out_fps = float(args.fps)

    tip = dots[:, -1]
    travel = (tip.max(0) - tip.min(0)) * 100
    print(f"[preview] {len(dots)} frames @ {out_fps:.0f}fps; flange travel cm "
          f"x={travel[0]:.1f} y={travel[1]:.1f} z={travel[2]:.1f}", flush=True)

    fig = plt.figure(figsize=(6, 6), dpi=100)   # -> 600x600, known for side-by-side
    ax = fig.add_subplot(111, projection="3d")
    # frame the arm from its actual reach (mimic swings much wider than the box)
    allpts = dots.reshape(-1, 3)
    ctr = allpts.mean(0)
    rad = max(0.5, float(np.abs(allpts - ctr).max()) * 1.1)
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(min(0, allpts[:, 2].min()), max(allpts[:, 2].max() * 1.1, rad))
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=12, azim=-72)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    if args.mode != "mimic":
        for a, b in _box_edges(BOX_CENTER, BOX_HALF):
            ax.plot(*zip(a, b), color="tab:orange", lw=0.6, alpha=0.5)

    ax.plot([], [], [], "-", color="0.75", lw=1)                 # (faint backbone, static)
    # colored chain segments (torso/upper/fore/hand) matching the stills
    seg_lines = [ax.plot([], [], [], "-o", color=c, lw=4, ms=5, mfc="white")[0]
                 for _, _, c in ROBOT_SEGMENTS]
    trail, = ax.plot([], [], [], "-", color="0.4", lw=1, alpha=0.4)
    hand_lines = ([ax.plot([], [], [], color="#ff7f0e", lw=2)[0] for _ in range(4)]
                  + [ax.plot([], [], [], color="darkorange", lw=3)[0]]) if flanges is not None else []
    title = ax.set_title("")

    def update(i):
        d = dots[i]
        for ln, (s, e, _) in zip(seg_lines, ROBOT_SEGMENTS):
            ln.set_data(d[s:e + 1, 0], d[s:e + 1, 1]); ln.set_3d_properties(d[s:e + 1, 2])
        lo = max(0, i - 30)
        tp = dots[lo:i + 1, -1]
        trail.set_data(tp[:, 0], tp[:, 1]); trail.set_3d_properties(tp[:, 2])
        if flanges is not None:
            for ln, (a, b) in zip(hand_lines, _hand_lines(flanges[i])):
                ln.set_data([a[0], b[0]], [a[1], b[1]]); ln.set_3d_properties([a[2], b[2]])
        title.set_text(f"{args.mode}:{label}  frame {i}/{len(dots)}")
        return (*seg_lines, trail, title)

    ani = FuncAnimation(fig, update, frames=len(dots), interval=1000 / out_fps, blit=False)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams["animation.ffmpeg_path"] = ff
    silent = args.out + ".silent.mp4"
    ani.save(silent, writer=FFMpegWriter(fps=out_fps, bitrate=2400))
    _finish(ff, silent, args.song, args.out, args.start, out_fps, args.dancer)
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
