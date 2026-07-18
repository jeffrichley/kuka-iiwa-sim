"""Spectral-residual saliency (Hou & Zhang, 2007) — pure numpy, no deep model."""
import numpy as np


def _resize(img, hw):
    """Nearest-neighbor resize to (H, W)."""
    img = np.asarray(img, float)
    H, W = hw
    ys = np.linspace(0, img.shape[0] - 1, H).astype(int)
    xs = np.linspace(0, img.shape[1] - 1, W).astype(int)
    return img[np.ix_(ys, xs)]


def _boxblur(a, k=3):
    """Separable-equivalent k x k box blur with edge padding."""
    a = np.asarray(a, float)
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a, dtype=float)
    for dy in range(k):
        for dx in range(k):
            out += ap[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / (k * k)


def spectral_residual(gray, out_hw=(64, 64)):
    """Estimate a saliency map from image structure. Returns (H,W) in [0,1]."""
    g = _resize(gray, out_hw)
    F = np.fft.fft2(g)
    log_amp = np.log(np.abs(F) + 1e-8)
    phase = np.angle(F)
    residual = log_amp - _boxblur(log_amp, 3)
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = _boxblur(sal, 3)
    lo, hi = float(sal.min()), float(sal.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(sal)
    return (sal - lo) / (hi - lo)
