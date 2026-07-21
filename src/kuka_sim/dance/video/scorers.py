"""Four pluggable saliency scorers: PoseTrack(+ctx) -> (F,2) focus points."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from kuka_sim.dance.video.saliency import spectral_residual

CENTER_LANDMARKS = [11, 12, 23, 24]   # shoulders + hips


@dataclass
class ScorerCtx:
    beat_env: Optional[np.ndarray] = None      # (F,) music energy on the frame grid
    gray_frames: Optional[np.ndarray] = None   # (F,h,w) grayscale frames


def _smooth_time(a, win):
    """Moving average along axis 0 (edge-padded, length-preserving)."""
    a = np.asarray(a, float)
    if win <= 1 or len(a) < win:
        return a
    pad_l = win // 2
    pad_r = win - 1 - pad_l
    ap = np.pad(a, [(pad_l, pad_r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    k = np.ones(win) / win
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, ap)


def _speed(xy):
    """(F,K) per-landmark frame-to-frame speed."""
    d = np.zeros(xy.shape[:2])
    d[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    return d


def _energy_scores(track):
    """(F,K) motion energy = speed * reach, time-smoothed over ~1s."""
    xy = track.xy
    center = xy[:, CENTER_LANDMARKS].mean(axis=1)            # (F,2)
    reach = np.linalg.norm(xy - center[:, None, :], axis=-1)  # (F,K)
    win = max(1, int(round(track.fps)))
    return _smooth_time(_speed(xy) * reach, win)


def _music_scores(track, beat_env):
    """(F,K) 'moving on the beat' = speed * beat_env, time-smoothed."""
    beat_env = np.asarray(beat_env, float)
    win = max(1, int(round(track.fps)))
    return _smooth_time(_speed(track.xy) * beat_env[:, None], win)


def _norm01_perframe(s):
    lo = s.min(axis=1, keepdims=True)
    hi = s.max(axis=1, keepdims=True)
    return np.divide(s - lo, hi - lo, out=np.zeros_like(s), where=(hi - lo) > 1e-12)


def _winner_focus(track, scores):
    """Per frame, the xy of the highest-scoring landmark."""
    win_idx = np.argmax(scores, axis=1)                      # (F,)
    return track.xy[np.arange(len(win_idx)), win_idx]        # (F,2)


def energy(track, ctx):
    return _winner_focus(track, _energy_scores(track))


def music(track, ctx):
    if ctx.beat_env is None:
        raise ValueError("music scorer requires ctx.beat_env")
    return _winner_focus(track, _music_scores(track, ctx.beat_env))


def saliency(track, ctx):
    if ctx.gray_frames is None:
        raise ValueError("saliency scorer requires ctx.gray_frames")
    frames = ctx.gray_frames
    F, K = track.xy.shape[:2]
    scores = np.zeros((F, K))
    for f in range(F):
        sm = spectral_residual(frames[f], out_hw=(64, 64))
        h, w = sm.shape
        px = np.clip((track.xy[f, :, 0] * (w - 1)).astype(int), 0, w - 1)
        py = np.clip((track.xy[f, :, 1] * (h - 1)).astype(int), 0, h - 1)
        scores[f] = sm[py, px]
    return _winner_focus(track, scores)


def blend(track, ctx):
    if ctx.beat_env is None:
        raise ValueError("blend scorer requires ctx.beat_env")
    e = _norm01_perframe(_energy_scores(track))
    m = _norm01_perframe(_music_scores(track, ctx.beat_env))
    return _winner_focus(track, 0.6 * e + 0.4 * m)


SCORERS = {"energy": energy, "music": music, "saliency": saliency, "blend": blend}
