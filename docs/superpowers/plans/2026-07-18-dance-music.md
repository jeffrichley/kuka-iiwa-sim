# V1 Music Choreography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn any song into an EE-space `DanceTrajectory` so the shipped dance
backend can make the iiwa 7 R800 dance to it and render an MP4 with the song.

**Architecture:** A new pure-logic front-end under `src/kuka_sim/dance/music/`:
`features.py` (librosa → `FeatureTrack` on the sim grid), `moves.py` (named
EE-space gesture generators), `choreographer.py` (`FeatureTrack` → `DanceTrajectory`).
A CLI `scripts/dance_music_demo.py` wires it to the backend's `play` + `record`.
All music math is pure numpy/librosa and unit-tested; only the render is smoke-tested.

**Tech Stack:** Python 3.11, numpy, librosa (new `[dance]` extra), pytest, the
existing `kuka_sim.dance` backend (`DanceTrajectory`, `map_to_workspace`,
`smooth_and_limit`, `FORWARD_QUAT`, `play`, `record`).

## Global Constraints

- Target venv: `env_isaaclab/Scripts/python.exe` (Python 3.11.9). Run pytest with
  `env_isaaclab/Scripts/python.exe -m pytest`.
- Sim dt is `1/120` s. `FeatureTrack.n == ceil(duration/dt)`; sample `i` ⇔ song
  time `i*dt`.
- Motion is authored in **EE space** (metre offsets in a workspace box), never
  joint space. Reuse `map_to_workspace` / `smooth_and_limit` from
  `kuka_sim.dance.trajectory` — do not reimplement them.
- Backend contract is fixed (`src/kuka_sim/dance/trajectory.py`):
  `DanceTrajectory(ee_pos, ee_quat=None, dt=1/120, cam_pos=None, cam_target=None,
  audio_path=None, pip_video_path=None)`; `FORWARD_QUAT = np.array([0.7071, 0.0,
  0.7071, 0.0])` (wxyz).
- Default workspace box: `center=(0.5, 0.0, 0.7)`, `half_extents=(0.10, 0.18, 0.14)`.
- `ee_quat` rows must be unit-norm quaternions (wxyz).
- No new deps beyond `librosa` (added to an optional `[dance]` extra); ffmpeg
  already ships via `imageio-ffmpeg`.
- Tests are flat under `tests/` named `test_dance_music_*.py`, matching the
  existing `test_dance_*.py` convention.
- Commit style: conventional commits; stage specific files; end messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `[dance]` extra + music subpackage scaffold

**Files:**
- Modify: `pyproject.toml:14-16` (optional-dependencies)
- Create: `src/kuka_sim/dance/music/__init__.py`
- Create: `tests/test_dance_music_pkg.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `kuka_sim.dance.music`; `librosa` installed in the
  venv for later tasks.

- [ ] **Step 1: Add the `dance` extra to pyproject**

In `pyproject.toml`, under `[project.optional-dependencies]`, add the `dance`
line so the block reads:

```toml
[project.optional-dependencies]
dev = ["pytest>=7.4"]
docs = ["mkdocs-material>=9.5"]
dance = ["librosa>=0.10"]
```

- [ ] **Step 2: Install librosa into the venv**

Run: `env_isaaclab/Scripts/python.exe -m pip install "librosa>=0.10"`
Expected: installs librosa + deps (numba, soundfile, audioread, …), ends with
`Successfully installed … librosa-0.1x.x …`.

- [ ] **Step 3: Create the package init**

Create `src/kuka_sim/dance/music/__init__.py`:

```python
"""V1 music choreography: song -> FeatureTrack -> DanceTrajectory."""
```

- [ ] **Step 4: Write the failing import test**

Create `tests/test_dance_music_pkg.py`:

```python
def test_music_package_imports():
    import kuka_sim.dance.music  # noqa: F401


def test_librosa_available():
    import librosa  # noqa: F401
    assert hasattr(librosa, "beat")
```

- [ ] **Step 5: Run the test**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_pkg.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/kuka_sim/dance/music/__init__.py tests/test_dance_music_pkg.py
git commit -m "feat(dance): add [dance] extra (librosa) + music subpackage scaffold"
```

---

### Task 2: `features.py` pure helpers

**Files:**
- Create: `src/kuka_sim/dance/music/features.py`
- Create: `tests/test_dance_music_features.py`

**Interfaces:**
- Consumes: numpy only.
- Produces (used by Task 3's `analyze`):
  - `normalize01(x: np.ndarray) -> np.ndarray` — min-max to [0,1]; constant input → all-zeros.
  - `impulses_to_grid(event_times: np.ndarray, n: int, dt: float) -> np.ndarray` —
    (n,) 0/1 array, 1 at `round(t/dt)` for each event time in range.
  - `resample_to_grid(values: np.ndarray, times: np.ndarray, n: int, dt: float) -> np.ndarray` —
    linear-interp `values` sampled at `times` onto grid `[0, dt, …, (n-1)dt]`.
  - `split_bands(mel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` —
    mel spectrogram `(n_mels, T)` → three per-frame energy series (bass/mid/treble),
    each the mean over its third of the mel bins; returns `(T,)` arrays.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dance_music_features.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_features.py -v`
Expected: FAIL (module `features` not found).

- [ ] **Step 3: Implement the helpers**

Create `src/kuka_sim/dance/music/features.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_features.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/music/features.py tests/test_dance_music_features.py
git commit -m "feat(dance): music feature DSP helpers (normalize/impulse/resample/bands)"
```

---

### Task 3: `FeatureTrack` + `analyze`

**Files:**
- Modify: `src/kuka_sim/dance/music/features.py` (append)
- Modify: `tests/test_dance_music_features.py` (append)

**Interfaces:**
- Consumes: Task 2 helpers; librosa.
- Produces (used by Task 5):
  - `FeatureTrack` dataclass with fields `dt, n, rms, bass, mid, treble, beats,
    onsets, tempo` (all `(n,)` arrays except `dt: float`, `n: int`, `tempo: float`).
  - `_load_audio(path: str) -> tuple[np.ndarray, int]` — thin seam returning
    `(y, sr)`; tests monkeypatch it.
  - `analyze(audio_path: str, dt: float = 1/120) -> FeatureTrack`.

- [ ] **Step 1: Write the failing test (monkeypatched load, real librosa math)**

Append to `tests/test_dance_music_features.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_features.py::test_analyze_produces_grid_aligned_track -v`
Expected: FAIL (`analyze` / `_load_audio` not defined).

- [ ] **Step 3: Implement `FeatureTrack`, `_load_audio`, `analyze`**

Append to `src/kuka_sim/dance/music/features.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_features.py -v`
Expected: all passed (Task 2 tests + the new analyze test).

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/music/features.py tests/test_dance_music_features.py
git commit -m "feat(dance): FeatureTrack + analyze (librosa -> sim-grid features)"
```

---

### Task 4: `moves.py` EE-space gesture generators

**Files:**
- Create: `src/kuka_sim/dance/music/moves.py`
- Create: `tests/test_dance_music_moves.py`

**Interfaces:**
- Consumes: numpy only.
- Produces (used by Task 5), each returning an `(n, 3)` offset in metres on box
  axes (x=depth, y=lateral, z=height):
  - `sway(t, freq, amp)` — lateral (y) sine.
  - `bob(t, freq, amp)` — vertical (z) sine.
  - `reach(t, env, amp)` — depth (x) = `amp*env` (env is `(n,)` in [0,1]).
  - `circle(t, freq, amp)` — y=cos, z=sin.
  - `figure_eight(t, freq, amp)` — y=sin(2πft), z=sin(4πft).
  - `accent(impulse, decay, amp, axis)` — exponentially-decaying kick after each
    impulse sample on `axis` (0=x,1=y,2=z), via causal exp-kernel convolution.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dance_music_moves.py`:

```python
import numpy as np
from kuka_sim.dance.music import moves


def _t(n=240, dt=1/120):
    return np.arange(n) * dt


def test_sway_is_lateral_only_and_bounded():
    off = moves.sway(_t(), freq=1.0, amp=0.05)
    assert off.shape == (240, 3)
    assert np.allclose(off[:, 0], 0.0) and np.allclose(off[:, 2], 0.0)
    assert np.abs(off[:, 1]).max() <= 0.05 + 1e-9


def test_bob_is_vertical_only():
    off = moves.bob(_t(), freq=2.0, amp=0.03)
    assert np.allclose(off[:, 0], 0.0) and np.allclose(off[:, 1], 0.0)
    assert np.abs(off[:, 2]).max() <= 0.03 + 1e-9


def test_reach_follows_envelope_on_x():
    env = np.linspace(0, 1, 240)
    off = moves.reach(_t(), env=env, amp=0.08)
    assert np.allclose(off[:, 1], 0.0) and np.allclose(off[:, 2], 0.0)
    assert np.isclose(off[-1, 0], 0.08)
    assert np.isclose(off[0, 0], 0.0)


def test_circle_traces_yz_circle():
    off = moves.circle(_t(), freq=1.0, amp=0.04)
    r = np.sqrt(off[:, 1] ** 2 + off[:, 2] ** 2)
    assert np.allclose(r, 0.04, atol=1e-9)
    assert np.allclose(off[:, 0], 0.0)


def test_figure_eight_shape():
    off = moves.figure_eight(_t(), freq=1.0, amp=0.04)
    assert off.shape == (240, 3)
    assert np.allclose(off[:, 0], 0.0)
    assert np.abs(off[:, 1]).max() <= 0.04 + 1e-9


def test_accent_decays_after_impulse():
    imp = np.zeros(240); imp[10] = 1.0
    off = moves.accent(imp, decay=0.3, amp=0.05, axis=2)
    assert np.allclose(off[:10, 2], 0.0)          # nothing before the impulse
    assert np.isclose(off[10, 2], 0.05)            # peak at the impulse
    assert off[11, 2] < off[10, 2]                 # decays afterward
    assert np.allclose(off[:, 0], 0.0) and np.allclose(off[:, 1], 0.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_moves.py -v`
Expected: FAIL (module `moves` not found).

- [ ] **Step 3: Implement the generators**

Create `src/kuka_sim/dance/music/moves.py`:

```python
"""EE-space gesture generators. Each returns an (n,3) metre offset on box axes
(x=depth, y=lateral, z=height). Pure numpy; summed + modulated by the
choreographer, then mapped into the box and speed-limited by the backend."""
import numpy as np

TWO_PI = 2.0 * np.pi


def _zeros(n):
    return np.zeros((n, 3), float)


def sway(t, freq, amp):
    t = np.asarray(t, float)
    off = _zeros(len(t))
    off[:, 1] = amp * np.sin(TWO_PI * freq * t)
    return off


def bob(t, freq, amp):
    t = np.asarray(t, float)
    off = _zeros(len(t))
    off[:, 2] = amp * np.sin(TWO_PI * freq * t)
    return off


def reach(t, env, amp):
    t = np.asarray(t, float)
    env = np.asarray(env, float)
    off = _zeros(len(t))
    off[:, 0] = amp * env
    return off


def circle(t, freq, amp):
    t = np.asarray(t, float)
    off = _zeros(len(t))
    off[:, 1] = amp * np.cos(TWO_PI * freq * t)
    off[:, 2] = amp * np.sin(TWO_PI * freq * t)
    return off


def figure_eight(t, freq, amp):
    t = np.asarray(t, float)
    off = _zeros(len(t))
    off[:, 1] = amp * np.sin(TWO_PI * freq * t)
    off[:, 2] = amp * np.sin(2.0 * TWO_PI * freq * t)
    return off


def accent(impulse, decay, amp, axis):
    """Causal exp-decay kick after each impulse: convolve the impulse train with
    amp*exp(-decay*k) for k>=0, then place on `axis`."""
    impulse = np.asarray(impulse, float)
    n = len(impulse)
    k = np.arange(n)
    kernel = amp * np.exp(-decay * k)
    conv = np.convolve(impulse, kernel)[:n]
    off = _zeros(n)
    off[:, axis] = conv
    return off
```

- [ ] **Step 4: Run to verify they pass**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_moves.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/music/moves.py tests/test_dance_music_moves.py
git commit -m "feat(dance): EE-space move library (sway/bob/reach/circle/fig8/accent)"
```

---

### Task 5: `choreographer.py` — FeatureTrack → DanceTrajectory

**Files:**
- Create: `src/kuka_sim/dance/music/choreographer.py`
- Create: `tests/test_dance_music_choreographer.py`

**Interfaces:**
- Consumes: `FeatureTrack` (Task 3), `moves` (Task 4),
  `kuka_sim.dance.trajectory.{DanceTrajectory, map_to_workspace, FORWARD_QUAT}`.
- Produces (used by Task 6):
  - `euler_to_quat(roll, pitch, yaw) -> np.ndarray` — wxyz unit quat (pure).
  - `quat_mul(a, b) -> np.ndarray` — Hamilton product, wxyz (pure).
  - `STYLES: dict[str, dict]` — parameter presets; keys include `"waltz"`, `"epic"`.
  - `choreograph(track, style="waltz", camera="hero",
    box_center=(0.5,0.0,0.7), box_half_extents=(0.10,0.18,0.14)) -> DanceTrajectory`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dance_music_choreographer.py`:

```python
import numpy as np
import pytest
from kuka_sim.dance.music.choreographer import (
    choreograph, euler_to_quat, quat_mul, STYLES,
)
from kuka_sim.dance.music.features import FeatureTrack


def _track(n=600, dt=1/120):
    rng = np.random.RandomState(0)
    beats = np.zeros(n); beats[::60] = 1.0
    onsets = np.zeros(n); onsets[::40] = 1.0
    ramp = np.linspace(0, 1, n)
    return FeatureTrack(
        dt=dt, n=n,
        rms=ramp, bass=np.abs(np.sin(np.linspace(0, 9, n))),
        mid=rng.rand(n), treble=rng.rand(n),
        beats=beats, onsets=onsets, tempo=120.0,
    )


def test_euler_to_quat_is_unit_and_identity_at_zero():
    q = euler_to_quat(0.0, 0.0, 0.0)
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])
    q2 = euler_to_quat(0.1, -0.2, 0.3)
    assert np.isclose(np.linalg.norm(q2), 1.0)


def test_quat_mul_identity():
    q = np.array([0.7071, 0.0, 0.7071, 0.0])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_mul(ident, q), q)


def test_choreograph_positions_inside_box():
    center = np.array([0.5, 0.0, 0.7]); half = np.array([0.10, 0.18, 0.14])
    traj = choreograph(_track(), style="waltz", camera="hero",
                       box_center=tuple(center), box_half_extents=tuple(half))
    assert traj.ee_pos.shape == (600, 3)
    assert np.all(traj.ee_pos <= center + half + 1e-6)
    assert np.all(traj.ee_pos >= center - half - 1e-6)


def test_choreograph_quats_unit_norm():
    traj = choreograph(_track(), style="waltz")
    assert traj.ee_quat.shape == (600, 4)
    norms = np.linalg.norm(traj.ee_quat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_hero_camera_is_none_cinematic_is_track():
    hero = choreograph(_track(), camera="hero")
    assert hero.cam_pos is None and hero.cam_target is None
    cine = choreograph(_track(), camera="cinematic")
    assert cine.cam_pos.shape == (600, 3)
    assert cine.cam_target.shape == (600, 3)


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        choreograph(_track(), style="nope")


def test_styles_have_required_presets():
    assert "waltz" in STYLES and "epic" in STYLES
```

- [ ] **Step 2: Run to verify they fail**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_choreographer.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the choreographer**

Create `src/kuka_sim/dance/music/choreographer.py`:

```python
"""FeatureTrack -> DanceTrajectory: combine band-modulated EE moves + a wrist
flourish + an optional camera track. Pure logic (no Isaac)."""
import numpy as np
from kuka_sim.dance.trajectory import DanceTrajectory, map_to_workspace, FORWARD_QUAT
from kuka_sim.dance.music import moves

TWO_PI = 2.0 * np.pi

# Per-style parameter presets. Amplitudes in metres / radians; freqs in Hz.
STYLES = {
    "waltz": dict(
        sway_freq=0.5, sway_amp=0.09, bob_freq=1.0, bob_amp=0.05,
        reach_amp=0.07, beat_amp=0.05, beat_decay=0.25,
        roll_amp=0.25, yaw_amp=0.20, wrist_freq=0.5, onset_amp=0.15,
        cam_orbit_freq=0.03, cam_radius=1.9, cam_push=0.5, cam_height=1.3,
    ),
    "epic": dict(
        sway_freq=0.25, sway_amp=0.10, bob_freq=0.5, bob_amp=0.06,
        reach_amp=0.10, beat_amp=0.06, beat_decay=0.15,
        roll_amp=0.30, yaw_amp=0.25, wrist_freq=0.3, onset_amp=0.20,
        cam_orbit_freq=0.02, cam_radius=2.1, cam_push=0.7, cam_height=1.4,
    ),
}


def euler_to_quat(roll, pitch, yaw):
    """Intrinsic XYZ euler -> wxyz unit quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z], float)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], float)


def choreograph(track, style="waltz", camera="hero",
                box_center=(0.5, 0.0, 0.7), box_half_extents=(0.10, 0.18, 0.14)):
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choices: {sorted(STYLES)}")
    p = STYLES[style]
    n, dt = track.n, track.dt
    t = np.arange(n) * dt
    center = np.asarray(box_center, float)
    half = np.asarray(box_half_extents, float)

    # --- position: band-modulated EE offsets summed in metres ---
    off = np.zeros((n, 3), float)
    off += moves.sway(t, p["sway_freq"], p["sway_amp"]) * track.bass[:, None]
    off += moves.bob(t, p["bob_freq"], p["bob_amp"]) * track.mid[:, None]
    off += moves.reach(t, track.rms, p["reach_amp"])
    off += moves.accent(track.beats, p["beat_decay"], p["beat_amp"], axis=2)
    # global swell: quiet passages small, peaks large
    off *= (0.5 + 0.5 * track.rms)[:, None]

    # normalize by half-extents so map_to_workspace clips into the box
    norm = np.divide(off, half, out=np.zeros_like(off), where=half != 0)
    ee_pos = map_to_workspace(norm, center, half)

    # --- orientation: bounded wrist flourish around FORWARD_QUAT ---
    roll = p["roll_amp"] * track.treble * np.sin(TWO_PI * p["wrist_freq"] * t)
    yaw = p["yaw_amp"] * track.treble * np.cos(TWO_PI * p["wrist_freq"] * t)
    onset_kick = moves.accent(track.onsets, p["beat_decay"], p["onset_amp"], axis=0)[:, 0]
    roll = roll + onset_kick
    ee_quat = np.array([
        quat_mul(euler_to_quat(roll[i], 0.0, yaw[i]), FORWARD_QUAT)
        for i in range(n)
    ])

    # --- camera ---
    cam_pos = cam_target = None
    if camera == "cinematic":
        theta = TWO_PI * p["cam_orbit_freq"] * t
        radius = p["cam_radius"] - p["cam_push"] * track.rms
        cam_pos = np.stack([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            np.full(n, p["cam_height"]),
        ], axis=1)
        cam_target = np.tile(center, (n, 1))
    elif camera != "hero":
        raise ValueError(f"unknown camera {camera!r}; choices: hero, cinematic")

    return DanceTrajectory(ee_pos=ee_pos, ee_quat=ee_quat, dt=dt,
                           cam_pos=cam_pos, cam_target=cam_target)
```

- [ ] **Step 4: Run to verify they pass**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_choreographer.py -v`
Expected: all passed.

- [ ] **Step 5: Run the full music suite**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_music_*.py -v`
Expected: all passed (pkg + features + moves + choreographer).

- [ ] **Step 6: Commit**

```bash
git add src/kuka_sim/dance/music/choreographer.py tests/test_dance_music_choreographer.py
git commit -m "feat(dance): choreographer (FeatureTrack -> DanceTrajectory) + quats + styles"
```

---

### Task 6: `dance_music_demo.py` CLI + end-to-end render

**Files:**
- Create: `scripts/dance_music_demo.py`

**Interfaces:**
- Consumes: `analyze`, `choreograph`, backend `play`/`record`/`smooth_and_limit`,
  `launch`, `build_scene`, `SceneCamera`.
- Produces: `out/<name>.mp4` with the song muxed. (Smoke-verified on GPU, not via
  a subagent — Isaac coupling per module convention.)

- [ ] **Step 1: Write the demo script**

Create `scripts/dance_music_demo.py`:

```python
"""V1: choreograph the arm to a song and render an MP4 (with the song muxed).

Usage:
    python scripts/dance_music_demo.py --song assets/audio/blue_danube.mp3 \
        --style waltz --camera hero --out out/blue_danube.mp4
    python scripts/dance_music_demo.py --song assets/audio/zarathustra.mp3 \
        --style epic --camera cinematic --seconds 20
"""
import argparse
import os
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.camera import SceneCamera
from kuka_sim.dance.trajectory import smooth_and_limit
from kuka_sim.dance.player import play
from kuka_sim.dance.recorder import record
from kuka_sim.dance.music.features import analyze
from kuka_sim.dance.music.choreographer import choreograph

USD = "assets/usd/iiwa7_r800.usd"
OUT_DIR = "out"
DT = 1.0 / 120.0
CAPTURE_EVERY = 4          # 120 Hz sim / 4 -> 30 fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--style", default="waltz")
    ap.add_argument("--camera", default="hero", choices=["hero", "cinematic"])
    ap.add_argument("--seconds", type=float, default=None,
                    help="render only the first N seconds (fast smoke)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[analyze] {args.song}", flush=True)
    track = analyze(args.song, dt=DT)
    traj = choreograph(track, style=args.style, camera=args.camera)

    if args.seconds is not None:
        cut = int(args.seconds / DT)
        traj.ee_pos = traj.ee_pos[:cut]
        traj.ee_quat = traj.ee_quat[:cut]
        if traj.cam_pos is not None:
            traj.cam_pos = traj.cam_pos[:cut]
            traj.cam_target = traj.cam_target[:cut]

    traj.ee_pos = smooth_and_limit(traj.ee_pos, DT, max_speed=0.6)
    traj.audio_path = args.song
    print(f"[choreograph] {len(traj)} samples, tempo={track.tempo:.1f} bpm", flush=True)

    app, sim = launch(headless=True, enable_cameras=True)
    handles = build_scene(sim, USD, with_probe=False, with_surface=False)
    cam = SceneCamera(pos=(1.9, 1.9, 1.3), target=(0.5, 0.0, 0.7))
    sim.reset(); handles["arm"].initialize(); cam.initialize()

    frames = []

    def on_step(i):
        if i % CAPTURE_EVERY == 0:
            frames.append(cam.capture())

    play(sim, handles, traj, cam=cam, on_step=on_step)

    name = os.path.splitext(os.path.basename(args.song))[0]
    out = args.out or os.path.join(OUT_DIR, f"{name}.mp4")
    record(frames, out, fps=30, audio_path=traj.audio_path)
    print(f"[OK] wrote {out} ({len(frames)} frames)", flush=True)
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-render a short Blue Danube clip (controller runs this directly)**

Run: `env_isaaclab/Scripts/python.exe scripts/dance_music_demo.py --song assets/audio/blue_danube.mp3 --style waltz --camera hero --seconds 12 --out out/blue_danube_smoke.mp4`
Expected: prints `[analyze]`, `[choreograph] … bpm`, then `[OK] wrote out/blue_danube_smoke.mp4 (~90 frames)`; exit 0.

- [ ] **Step 3: Verify the clip has video + audio streams**

Run: `env_isaaclab/Scripts/python.exe -c "import imageio_ffmpeg,subprocess; ff=imageio_ffmpeg.get_ffmpeg_exe(); print(subprocess.run([ff,'-i','out/blue_danube_smoke.mp4'],capture_output=True).stderr.decode('utf-8','replace'))"`
Expected: output lists a `Video: h264` stream and an `Audio: aac` stream.

- [ ] **Step 4: Full renders — both showcase songs**

Run: `env_isaaclab/Scripts/python.exe scripts/dance_music_demo.py --song assets/audio/blue_danube.mp3 --style waltz --camera hero --out out/blue_danube.mp4`
Run: `env_isaaclab/Scripts/python.exe scripts/dance_music_demo.py --song assets/audio/zarathustra.mp3 --style epic --camera cinematic --out out/zarathustra.mp4`
Expected: each prints `[OK] wrote …`; both MP4s play with the song and the arm dances in time.

- [ ] **Step 5: Commit**

```bash
git add scripts/dance_music_demo.py
git commit -m "feat(dance): music dance demo CLI (song -> choreograph -> render+audio)"
```

---

## Self-Review

**Spec coverage:** features (Task 2–3), moves (Task 4), choreographer + styles +
camera (Task 5), CLI + both showcase renders (Task 6), `[dance]`/librosa extra
(Task 1). All spec deliverables A–D covered.

**Placeholder scan:** none — every step carries full code or an exact command.

**Type consistency:** `FeatureTrack` fields defined in Task 3 are consumed by
name in Task 5; `choreograph`/`analyze` signatures match the spec and the demo;
backend names (`DanceTrajectory`, `map_to_workspace`, `smooth_and_limit`,
`FORWARD_QUAT`, `play`, `record`) match `trajectory.py`/`player.py`/`recorder.py`.
