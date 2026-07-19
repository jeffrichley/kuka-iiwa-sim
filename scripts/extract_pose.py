"""Standalone pose extraction — runs in the ISOLATED .pose_venv, NOT the Isaac venv.

MediaPipe pulls opencv-contrib, which would clobber Isaac's cv2, so pose lives in
its own venv and hands off a plain .npz (xy, visible, fps, gray) that the Isaac
render process loads. This script is self-contained (no kuka_sim import) so it
runs with only mediapipe + imageio + numpy.

Usage (pose venv):
    .pose_venv/Scripts/python.exe scripts/extract_pose.py \
        --video assets/video/dance.mp4 --out out/dance_poses.npz
"""
import argparse
import numpy as np

N_LANDMARKS = 33
GRAY_HW = (64, 64)


def _resize_gray(rgb, hw=GRAY_HW):
    """Grayscale + nearest-neighbor downsample a frame to hw (for saliency)."""
    g = np.asarray(rgb, float)[:, :, :3].mean(axis=2)
    H, W = hw
    ys = np.linspace(0, g.shape[0] - 1, H).astype(int)
    xs = np.linspace(0, g.shape[1] - 1, W).astype(int)
    return g[np.ix_(ys, xs)]


def read_frames(video_path, max_frames=None):
    # imageio's ffmpeg reader (imageio-ffmpeg) reliably reports true fps.
    import imageio
    r = imageio.get_reader(video_path)
    fps = float(r.get_meta_data().get("fps", 30.0))
    frames = []
    for i, fr in enumerate(r):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(np.asarray(fr)[:, :, :3])
    r.close()
    return frames, fps


def detect(frames):
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)
    F = len(frames)
    xy = np.full((F, N_LANDMARKS, 2), np.nan)
    xyz = np.full((F, N_LANDMARKS, 3), np.nan)   # 3D world landmarks (metres, hip origin)
    vis = np.zeros((F, N_LANDMARKS))
    for i, fr in enumerate(frames):
        res = pose.process(np.ascontiguousarray(fr))
        if res.pose_landmarks:
            for k, lm in enumerate(res.pose_landmarks.landmark):
                xy[i, k] = [lm.x, lm.y]
                vis[i, k] = lm.visibility
        if res.pose_world_landmarks:
            for k, lm in enumerate(res.pose_world_landmarks.landmark):
                xyz[i, k] = [lm.x, lm.y, lm.z]
    pose.close()
    return xy, xyz, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    print(f"[read] {args.video}", flush=True)
    frames, fps = read_frames(args.video, args.max_frames)
    print(f"[detect] {len(frames)} frames @ {fps:.2f} fps", flush=True)
    xy, xyz, vis = detect(frames)
    gray = np.stack([_resize_gray(f) for f in frames]) if frames else np.zeros((0,) + GRAY_HW)
    np.savez_compressed(args.out, xy=xy, xyz=xyz, visible=vis, fps=fps, gray=gray)
    detected = int((~np.isnan(xy[:, 0, 0])).sum())
    print(f"[OK] wrote {args.out}: {len(frames)} frames, pose found in {detected}", flush=True)


if __name__ == "__main__":
    main()
