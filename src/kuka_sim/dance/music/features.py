"""Audio -> FeatureTrack on the sim grid (librosa) + pure DSP helpers."""
from dataclasses import dataclass
import numpy as np


def normalize01(x):
    """Min-max normalize to [0, 1]. Constant input -> all zeros."""
    x = np.asarray(x, float)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def impulses_to_grid(event_times, n, dt):
    """(n,) 0/1 array with 1 at the nearest sim sample for each event time."""
    grid = np.zeros(n, float)
    for t in np.asarray(event_times, float):
        idx = int(round(float(t) / dt))
        if 0 <= idx < n:
            grid[idx] = 1.0
    return grid


def resample_to_grid(values, times, n, dt):
    """Linear-interpolate a (T,) series sampled at `times` onto [0..(n-1)dt]."""
    values = np.asarray(values, float)
    times = np.asarray(times, float)
    grid_t = np.arange(n) * dt
    return np.interp(grid_t, times, values)


def split_bands(mel):
    """Mel spectrogram (n_mels, T) -> (bass, mid, treble) per-frame means."""
    mel = np.asarray(mel, float)
    n_mels = mel.shape[0]
    b0, b1 = n_mels // 3, 2 * n_mels // 3
    bass = mel[:b0].mean(axis=0)
    mid = mel[b0:b1].mean(axis=0)
    treble = mel[b1:].mean(axis=0)
    return bass, mid, treble


@dataclass
class FeatureTrack:
    dt: float
    n: int
    rms: np.ndarray
    bass: np.ndarray
    mid: np.ndarray
    treble: np.ndarray
    beats: np.ndarray
    onsets: np.ndarray
    tempo: float


def _load_audio(path):
    """Load an audio file to mono float32 + sample rate. Thin seam for tests."""
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def analyze(audio_path, dt=1.0 / 120.0):
    """Analyze a song into a FeatureTrack sampled on the sim grid (n=ceil(dur/dt))."""
    import librosa
    y, sr = _load_audio(audio_path)
    duration = len(y) / float(sr)
    n = int(np.ceil(duration / dt))

    hop = 512
    frame_times = librosa.frames_to_time(
        np.arange(1 + len(y) // hop), sr=sr, hop_length=hop)

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=48)
    bass_f, mid_f, treble_f = split_bands(mel)

    def grid(series):
        m = min(len(series), len(frame_times))
        return resample_to_grid(series[:m], frame_times[:m], n, dt)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    return FeatureTrack(
        dt=dt, n=n,
        rms=normalize01(grid(rms)),
        bass=normalize01(grid(bass_f)),
        mid=normalize01(grid(mid_f)),
        treble=normalize01(grid(treble_f)),
        beats=impulses_to_grid(beat_times, n, dt),
        onsets=impulses_to_grid(onset_times, n, dt),
        tempo=float(np.atleast_1d(tempo)[0]),
    )
