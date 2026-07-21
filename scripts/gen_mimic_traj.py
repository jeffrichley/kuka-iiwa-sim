"""Bake the two-arm mimic joint trajectories for the Isaac render.

Runs in .preview_venv (mimic needs ikpy/scipy). Loads a pose .npz, solves the
right + left arm joint angles per frame, smooths, resamples to the sim rate, and
saves a plain .npz the Isaac render loads (no ikpy/scipy needed on the Isaac side).

Usage:
    .preview_venv/Scripts/python.exe scripts/gen_mimic_traj.py \
        --poses out/yt1_full_poses.npz --start 20 --seconds 40 \
        --sim-fps 60 --out out/yt1_traj.npz
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
from kuka_sim.dance.video.mimic import mimic_joint_traj


def _smooth(q, win):
    if win <= 1 or len(q) < win:
        return q
    pad_l, pad_r = win // 2, win - 1 - win // 2
    qp = np.pad(q, ((pad_l, pad_r), (0, 0)), mode="edge")
    k = np.ones(win) / win
    return np.stack([np.convolve(qp[:, j], k, mode="valid") for j in range(q.shape[1])], axis=1)


def _resample(q, src_fps, dst_fps):
    F = len(q)
    dur = F / src_fps
    n = int(round(dur * dst_fps))
    src_t = np.arange(F) / src_fps
    dst_t = np.arange(n) / dst_fps
    return np.stack([np.interp(dst_t, src_t, q[:, j]) for j in range(q.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--sim-fps", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    d = np.load(args.poses)
    xyz = np.nan_to_num(d["xyz"].astype(float), nan=0.0)
    fps = float(d["fps"])
    s0 = int(args.start * fps)
    xyz = xyz[s0:]
    if args.seconds is not None:
        xyz = xyz[:int(args.seconds * fps)]
    print(f"[gen] {len(xyz)} pose frames @ {fps:.2f}fps -> sim {args.sim_fps:.0f}fps", flush=True)

    win = max(1, int(round(fps * 0.15)))
    out = {}
    for sd in ("right", "left"):
        q = _smooth(mimic_joint_traj(xyz, side=sd), win)
        out[sd] = _resample(q, fps, args.sim_fps)
        print(f"[gen] {sd}: {out[sd].shape}", flush=True)

    np.savez(args.out, qr=out["right"], ql=out["left"], sim_fps=args.sim_fps,
             start=args.start, seconds=args.seconds if args.seconds else len(xyz) / fps)
    print(f"[OK] wrote {args.out} ({out['right'].shape[0]} sim frames)", flush=True)


if __name__ == "__main__":
    main()
