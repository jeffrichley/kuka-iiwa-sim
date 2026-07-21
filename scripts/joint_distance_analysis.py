"""How well does the robot track her joints over the whole video?

For each frame we take the mapped joint chain -- shoulder, elbow, wrist, hand --
for BOTH her (MediaPipe world landmarks) and the robot (mimic -> FK). To compare
across different frames/scales, both are anchored at the shoulder and normalized
by upper-arm length. Then we measure the distance between each robot joint and
her counterpart.

Outputs (preview venv):
  out/joint_distance_over_time.png  -- distance vs time for elbow/wrist/hand
  out/joint_overlay_3d.mp4          -- 3D overlay (her vs robot) + error lines

Usage:
    .preview_venv/Scripts/python.exe scripts/joint_distance_analysis.py \
        --poses out/dance_studio_poses.npz --side right --seconds 20
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

from kuka_sim.dance.video.mimic import mimic_joint_traj, _to_robot

URDF = "assets/urdf/lbr_iiwa7_r800_description/iiwa7_r800.urdf"
ROBOT_JOINTS = [2, 4, 6, 8]       # FK dots: shoulder, elbow, wrist, hand
NAMES = ["shoulder", "elbow", "wrist", "hand"]
COLORS = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e"]


def _norm_chain(pts):
    """Anchor at the shoulder (pts[0]) and scale by upper-arm length |el-sh|."""
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[1]) + 1e-9
    return pts / scale


def robot_chain(xyz, side):
    chain = Chain.from_urdf_file(URDF, base_elements=["lbr_link_0"])
    q7 = mimic_joint_traj(xyz, side=side)
    out = []
    for q in q7:
        fk = chain.forward_kinematics(np.concatenate([[0.0], q, [0.0]]),
                                      full_kinematics=True)
        dots = np.array([T[:3, 3] for T in fk])
        r = dots[ROBOT_JOINTS]                        # robot z-up already
        out.append(_norm_chain(r))
    return np.array(out)                              # (F,4,3)


def human_chain(xyz, side):
    sh, el, wr = (12, 14, 16) if side == "right" else (11, 13, 15)
    idx, pky = (20, 18) if side == "right" else (19, 17)
    out = []
    for p in xyz:
        hand = (p[idx] + p[pky]) / 2.0
        # convert to the ROBOT frame (same rotation the mapping targets) so the
        # comparison is apples-to-apples.
        H = np.array([_to_robot(v) for v in (p[sh], p[el], p[wr], hand)])
        out.append(_norm_chain(H))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="out/dance_studio_poses.npz")
    ap.add_argument("--side", default="right")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    d = np.load(args.poses)
    xyz = np.nan_to_num(d["xyz"].astype(float), nan=0.0)
    fps = float(d["fps"])
    if args.seconds is not None:
        xyz = xyz[:int(args.seconds * fps)]

    R = robot_chain(xyz, args.side)
    H = human_chain(xyz, args.side)
    dist = np.linalg.norm(R - H, axis=2)              # (F,4) per-joint distance
    t = np.arange(len(xyz)) / fps

    # --- 1) distance over time ---
    fig1, ax1 = plt.subplots(figsize=(11, 4), dpi=110)
    for j in (1, 2, 3):
        ax1.plot(t, dist[:, j], color=COLORS[j], lw=1.5,
                 label=f"{NAMES[j]} (mean {dist[:, j].mean():.2f})")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("distance (upper-arm lengths)")
    ax1.set_title("robot joint vs her joint — tracking distance over the dance")
    ax1.legend(); ax1.grid(alpha=0.3)
    fig1.tight_layout(); fig1.savefig("out/joint_distance_over_time.png")
    print("[OK] out/joint_distance_over_time.png "
          f"(overall mean {dist[:, 1:].mean():.2f} upper-arm lengths)", flush=True)

    # --- 2) 3D overlay animation ---
    fig2 = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig2.add_subplot(111, projection="3d")
    lim = 3.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=12, azim=-72)
    ax.set_xlabel("X (left/right)"); ax.set_ylabel("Y (depth)"); ax.set_zlabel("Z (up)")

    her_line, = ax.plot([], [], [], "-o", color="black", lw=3, ms=6, label="her")
    rob_line, = ax.plot([], [], [], "--s", color="crimson", lw=2, ms=6, label="robot")
    err_lines = [ax.plot([], [], [], color=COLORS[j], lw=1.5, alpha=0.7)[0] for j in range(4)]
    ax.legend(loc="upper left")
    txt = ax.set_title("")

    step = max(1, int(round(fps / args.fps)))
    frames = range(0, len(xyz), step)

    def update(i):
        h, r = H[i], R[i]
        her_line.set_data(h[:, 0], h[:, 1]); her_line.set_3d_properties(h[:, 2])
        rob_line.set_data(r[:, 0], r[:, 1]); rob_line.set_3d_properties(r[:, 2])
        for j, ln in enumerate(err_lines):
            ln.set_data([h[j, 0], r[j, 0]], [h[j, 1], r[j, 1]])
            ln.set_3d_properties([h[j, 2], r[j, 2]])
        txt.set_text(f"t={i/fps:4.1f}s   elbow {dist[i,1]:.2f}  wrist {dist[i,2]:.2f}  hand {dist[i,3]:.2f}")
        return (her_line, rob_line, *err_lines, txt)

    ani = FuncAnimation(fig2, update, frames=frames, interval=1000 / args.fps, blit=False)
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    ani.save("out/joint_overlay_3d.mp4", writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    print(f"[OK] out/joint_overlay_3d.mp4 ({len(list(frames))} frames)", flush=True)


if __name__ == "__main__":
    main()
