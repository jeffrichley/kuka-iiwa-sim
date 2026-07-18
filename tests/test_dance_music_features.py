import numpy as np
from kuka_sim.dance.music.features import (
    normalize01, impulses_to_grid, resample_to_grid, split_bands,
)


def test_normalize01_scales_to_unit_range():
    out = normalize01(np.array([2.0, 4.0, 6.0]))
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_normalize01_constant_input_is_zeros():
    out = normalize01(np.array([3.0, 3.0, 3.0]))
    assert np.allclose(out, 0.0)


def test_impulses_to_grid_marks_nearest_samples():
    # dt=0.5 -> grid times 0,0.5,1.0,1.5 ; events at 0.5 and 1.4 (->sample 3)
    g = impulses_to_grid(np.array([0.5, 1.4]), n=4, dt=0.5)
    assert g.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert set(np.unique(g)).issubset({0.0, 1.0})


def test_impulses_to_grid_ignores_out_of_range():
    g = impulses_to_grid(np.array([-1.0, 100.0]), n=4, dt=0.5)
    assert g.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_resample_to_grid_linear_interp():
    # values 0,10 at times 0,1 -> grid dt=0.5,n=3 -> 0,5,10
    out = resample_to_grid(np.array([0.0, 10.0]), np.array([0.0, 1.0]), n=3, dt=0.5)
    assert np.allclose(out, [0.0, 5.0, 10.0])


def test_split_bands_averages_thirds():
    # 6 mel bins, 2 frames; bins 0-1 bass, 2-3 mid, 4-5 treble
    mel = np.array([
        [1.0, 1.0], [1.0, 1.0],     # bass -> mean 1
        [2.0, 2.0], [2.0, 2.0],     # mid  -> mean 2
        [4.0, 4.0], [4.0, 4.0],     # treble -> mean 4
    ])
    bass, mid, treble = split_bands(mel)
    assert np.allclose(bass, 1.0)
    assert np.allclose(mid, 2.0)
    assert np.allclose(treble, 4.0)
    assert bass.shape == (2,)


def _click_track(sr=22050, seconds=4.0, bpm=120):
    """Synthesize a click train at a fixed tempo: a short noise burst per beat."""
    n = int(sr * seconds)
    y = np.zeros(n, float)
    step = int(sr * 60.0 / bpm)
    for start in range(0, n, step):
        end = min(start + 200, n)
        y[start:end] = np.random.RandomState(0).randn(end - start) * 0.5
    return y.astype(np.float32), sr


def test_analyze_produces_grid_aligned_track(monkeypatch):
    from kuka_sim.dance.music import features as F
    y, sr = _click_track(seconds=4.0, bpm=120)
    monkeypatch.setattr(F, "_load_audio", lambda p: (y, sr))

    dt = 1.0 / 120.0
    track = F.analyze("ignored.wav", dt=dt)

    expected_n = int(np.ceil(4.0 / dt))
    assert track.n == expected_n
    for arr in (track.rms, track.bass, track.mid, track.treble,
                track.beats, track.onsets):
        assert arr.shape == (expected_n,)
    # bands + rms normalized into [0,1]
    for arr in (track.rms, track.bass, track.mid, track.treble):
        assert arr.min() >= -1e-6 and arr.max() <= 1.0 + 1e-6
    # impulse trains are strictly 0/1 and non-empty for a 120bpm click track
    for arr in (track.beats, track.onsets):
        assert set(np.unique(arr)).issubset({0.0, 1.0})
    assert track.beats.sum() > 0
    assert track.tempo > 0
