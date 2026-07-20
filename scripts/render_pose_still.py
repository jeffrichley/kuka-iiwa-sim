"""Render a single still: the input photo (left) next to the iiwa posed by the
mimic mapping (right). A stationary side-by-side makes mapping reversals easy to
spot. Runs in .preview_venv (matplotlib + ikpy; loads an .npz from extract_pose).

Usage (preview venv):
    .preview_venv/Scripts/python.exe scripts/render_pose_still.py \
        --pose out/photo_pose.npz --photo assets/photos/pose1.jpg \
        --side right --out out/still_pose1.png
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
import matplotlib.image as mpimg
from ikpy.chain import Chain

from kuka_sim.dance.video.mimic import mimic_joint_traj

URDF = "assets/urdf/lbr_iiwa7_r800_description/iiwa7_r800.urdf"

# MediaPipe skeleton edges (subset that reads clearly) + the arm chains.
BODY_EDGES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15),
              (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28),
              (27, 31), (28, 32), (15, 17), (16, 18)]
ARM_CHAIN = {"right": [(12, 14), (14, 16)], "left": [(11, 13), (13, 15)]}


def _hand_glyph(ax, flange_T):
    """Draw an ORIENTED hand at the flange so its facing is legible: 4 fingers
    along the approach axis, spread across the palm axis, plus a distinct thumb
    along the palm-side axis (shows the wrist roll)."""
    ee = flange_T[:3, 3]
    approach = flange_T[:3, 2]   # flange z = where the hand points
    palm_x = flange_T[:3, 0]     # across the palm
    for s in (-1.5, -0.5, 0.5, 1.5):
        base = ee + palm_x * (s * 0.02)
        tip = base + approach * 0.09
        ax.plot(*[[base[k], tip[k]] for k in range(3)], color="crimson", lw=2.5)
    thumb = ee + palm_x * 0.06 + approach * 0.03
    ax.plot(*[[ee[k], thumb[k]] for k in range(3)], color="darkorange", lw=4)  # thumb = roll
    ax.plot([ee[0]], [ee[1]], [ee[2]], "o", color="crimson", ms=6)


def fk_dots(q7):
    chain = Chain.from_urdf_file(URDF, base_elements=["lbr_link_0"])
    q_full = np.concatenate([[0.0], q7, [0.0]])
    fk = chain.forward_kinematics(q_full, full_kinematics=True)
    dots = np.array([T[:3, 3] for T in fk])
    return dots, fk[-1]              # positions + flange transform (for orientation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help=".npz from extract_pose --image")
    ap.add_argument("--photo", required=True)
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--azim", type=float, default=-72.0)   # near-front, but 3/4 enough to show the torso fold
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    data = np.load(args.pose)
    xyz = np.nan_to_num(data["xyz"].astype(float), nan=0.0)
    xy = np.nan_to_num(data["xy"][0].astype(float), nan=0.0)   # 2D, normalized [0,1]
    q = mimic_joint_traj(xyz[:1], side=args.side)[0]
    dots, flange_T = fk_dots(q)
    names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    print("[still] joint deg: " +
          " ".join(f"{n}={np.rad2deg(v):+.0f}" for n, v in zip(names, q)), flush=True)

    fig = plt.figure(figsize=(11, 6), dpi=110)
    axp = fig.add_subplot(1, 2, 1)
    img = mpimg.imread(args.photo)
    H, W = img.shape[:2]
    axp.imshow(img); axp.axis("off")
    px, py = xy[:, 0] * W, xy[:, 1] * H
    for a, b in BODY_EDGES:                                   # full skeleton (green)
        axp.plot([px[a], px[b]], [py[a], py[b]], color="lime", lw=2, alpha=0.7, zorder=2)
    for a, b in ARM_CHAIN[args.side]:                        # driving arm (red)
        axp.plot([px[a], px[b]], [py[a], py[b]], color="red", lw=3.5, zorder=3)
    # her hand: wrist -> index / pinky / thumb (orange) so its facing is visible
    wr, idx, pky, thm = (16, 20, 18, 22) if args.side == "right" else (15, 19, 17, 21)
    for tip in (idx, pky, thm):
        axp.plot([px[wr], px[tip]], [py[wr], py[tip]], color="darkorange", lw=2.5, zorder=3)
    axp.scatter(px, py, s=14, c="yellow", edgecolors="black", lw=0.4, zorder=4)
    axp.set_title(f"photo + pose (red = {args.side} arm, orange = hand)")

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    allp = dots
    ctr = allp.mean(0); rad = max(0.5, float(np.abs(allp - ctr).max()) * 1.1)
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(min(0, allp[:, 2].min()), max(allp[:, 2].max() * 1.1, rad))
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.plot(dots[:, 0], dots[:, 1], dots[:, 2], "-o", color="tab:blue", lw=3, ms=6, mfc="white")
    _hand_glyph(ax, flange_T)
    ax.set_title(f"robot (mimic:{args.side})")

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[OK] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
