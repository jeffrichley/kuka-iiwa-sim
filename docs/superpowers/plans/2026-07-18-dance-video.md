# V2 Video → Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a dance video into an EE-space `DanceTrajectory` — estimate the
dancer's pose, pick the "most interesting" moving point each frame via one of
four pluggable scorers, and have the arm track it — then let the backend render
it with the source clip inset as picture-in-picture.

**Architecture:** New pure-logic front-end `src/kuka_sim/dance/video/`:
`saliency.py` (spectral-residual map), `pose.py` (MediaPipe → `PoseTrack`, seams
mocked in tests), `scorers.py` (energy/music/saliency/blend registry),
`retarget.py` (focus path → `DanceTrajectory`). CLI `scripts/dance_video_demo.py`
wires it to the backend. All math is unit-tested; only the render is smoke-tested.

**Tech Stack:** Python 3.11, numpy, mediapipe (new, `[dance]` extra), imageio,
librosa (already in `[dance]`), pytest, the `kuka_sim.dance` backend.

## Global Constraints

- Target venv: `env_isaaclab/Scripts/python.exe` (Python 3.11.9). Run pytest with
  `env_isaaclab/Scripts/python.exe -m pytest`.
- Sim dt is `1/120` s. The focus path is at video `fps`; resample to the sim grid
  (`N = round(F/fps/dt)`).
- Motion is authored in **EE space**; reuse `map_to_workspace`/`smooth_and_limit`
  and `FORWARD_QUAT` from `kuka_sim.dance.trajectory`. Build a
  `DanceTrajectory(ee_pos=…, dt=1/120, pip_video_path=video, audio_path=video)`.
- Image coords are normalized `[0,1]`, x→right, y→down. Map image-x → box lateral
  (**y**), `(1 − image-y)` → box height (**z**), depth (**x**) held at box center
  unless a depth envelope is given.
- Default workspace box: `center=(0.5,0.0,0.7)`, `half_extents=(0.10,0.18,0.14)`.
- MediaPipe Pose has **K=33** landmarks; body-center landmarks are L/R shoulder
  (11, 12) and L/R hip (23, 24).
- Saliency is **pure numpy** (spectral residual) — no opencv-contrib / deep model.
- No new deps beyond `mediapipe` (+ `imageio`, likely already present). ffmpeg via
  `imageio-ffmpeg`.
- Tests are flat under `tests/`, named `test_dance_video_*.py`.
- Commit style: conventional commits; stage specific files; end messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `mediapipe` dep + video subpackage scaffold

**Files:**
- Modify: `pyproject.toml` (the `dance` extra)
- Create: `src/kuka_sim/dance/video/__init__.py`
- Create: `tests/test_dance_video_pkg.py`

**Interfaces:**
- Produces: importable `kuka_sim.dance.video`; `mediapipe` + `imageio` in the venv.

- [ ] **Step 1: Extend the `dance` extra**

In `pyproject.toml`, change the `dance` line to:

```toml
dance = ["librosa>=0.10", "mediapipe>=0.10", "imageio>=2.31"]
```

- [ ] **Step 2: Install the new deps**

Run: `env_isaaclab/Scripts/python.exe -m pip install "mediapipe>=0.10" "imageio>=2.31"`
Expected: `Successfully installed … mediapipe-0.10.x …` (imageio likely already present).

- [ ] **Step 3: Create the package init**

Create `src/kuka_sim/dance/video/__init__.py`:

```python
"""V2 video->arm: dance video -> pose -> saliency scorer -> DanceTrajectory."""
```

- [ ] **Step 4: Write the failing import test**

Create `tests/test_dance_video_pkg.py`:

```python
def test_video_package_imports():
    import kuka_sim.dance.video  # noqa: F401


def test_mediapipe_available():
    import mediapipe  # noqa: F401
```

- [ ] **Step 5: Run the test**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_pkg.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/kuka_sim/dance/video/__init__.py tests/test_dance_video_pkg.py
git commit -m "feat(dance): add mediapipe/imageio to [dance] extra + video subpackage"
```

---

### Task 2: `saliency.py` — spectral-residual saliency

**Files:**
- Create: `src/kuka_sim/dance/video/saliency.py`
- Create: `tests/test_dance_video_saliency.py`

**Interfaces:**
- Produces (used by Task 4): `spectral_residual(gray, out_hw=(64,64)) -> (H,W)
  array in [0,1]`; helpers `_resize(img, hw)`, `_boxblur(a, k=3)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dance_video_saliency.py`:

```python
import numpy as np
from kuka_sim.dance.video.saliency import spectral_residual, _resize, _boxblur


def test_resize_nearest_shape():
    img = np.arange(100, dtype=float).reshape(10, 10)
    out = _resize(img, (5, 5))
    assert out.shape == (5, 5)


def test_boxblur_preserves_shape_and_smooths():
    a = np.zeros((8, 8)); a[4, 4] = 1.0
    out = _boxblur(a, 3)
    assert out.shape == (8, 8)
    assert out[4, 4] < 1.0            # energy spread to neighbors
    assert out[3, 4] > 0.0


def test_spectral_residual_is_unit_range():
    rng = np.random.RandomState(0)
    gray = rng.rand(48, 64)
    sm = spectral_residual(gray, out_hw=(32, 32))
    assert sm.shape == (32, 32)
    assert sm.min() >= 0.0 and sm.max() <= 1.0


def test_spectral_residual_highlights_localized_feature():
    # a small bright square on a flat background is salient vs the background
    gray = np.zeros((64, 64))
    gray[26:38, 26:38] = 1.0
    sm = spectral_residual(gray, out_hw=(64, 64))
    blob = sm[26:38, 26:38].mean()
    corner = sm[0:12, 0:12].mean()
    assert blob > corner
```

- [ ] **Step 2: Run to verify they fail**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_saliency.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/kuka_sim/dance/video/saliency.py`:

```python
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_saliency.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/video/saliency.py tests/test_dance_video_saliency.py
git commit -m "feat(dance): spectral-residual saliency map (pure numpy)"
```

---

### Task 3: `pose.py` — MediaPipe → `PoseTrack`

**Files:**
- Create: `src/kuka_sim/dance/video/pose.py`
- Create: `tests/test_dance_video_pose.py`

**Interfaces:**
- Produces (used by Tasks 4–5):
  - `PoseTrack` dataclass: `xy (F,K,2)`, `visible (F,K)`, `fps: float`.
  - `_fill_gaps(xy) -> xy` — per-landmark/coord linear interpolation over NaNs.
  - `_read_frames(path, max_frames) -> (frames, fps)` and
    `_detect_landmarks(frames) -> (xy_with_nan, visible)` — seams (mocked in tests).
  - `estimate_poses(video_path, max_frames=None) -> PoseTrack`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dance_video_pose.py`:

```python
import numpy as np
from kuka_sim.dance.video import pose as P
from kuka_sim.dance.video.pose import PoseTrack, _fill_gaps, estimate_poses


def test_fill_gaps_interpolates_interior_nan():
    xy = np.array([[[0.0, 0.0]], [[np.nan, np.nan]], [[2.0, 2.0]]])  # (3,1,2)
    out = _fill_gaps(xy)
    assert np.allclose(out[1, 0], [1.0, 1.0])       # interpolated midpoint
    assert not np.isnan(out).any()


def test_fill_gaps_all_nan_landmark_defaults_center():
    xy = np.full((4, 1, 2), np.nan)
    out = _fill_gaps(xy)
    assert np.allclose(out, 0.5)


def test_estimate_poses_assembles_track(monkeypatch):
    frames = [np.zeros((8, 8, 3), np.uint8)] * 5
    xy = np.full((5, 33, 2), 0.5); xy[2, 10] = [np.nan, np.nan]
    vis = np.ones((5, 33))
    monkeypatch.setattr(P, "_read_frames", lambda path, max_frames=None: (frames, 30.0))
    monkeypatch.setattr(P, "_detect_landmarks", lambda fr: (xy, vis))

    track = estimate_poses("ignored.mp4")
    assert isinstance(track, PoseTrack)
    assert track.xy.shape == (5, 33, 2)
    assert track.fps == 30.0
    assert not np.isnan(track.xy).any()             # gap filled
```

- [ ] **Step 2: Run to verify they fail**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_pose.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/kuka_sim/dance/video/pose.py`:

```python
"""Dance video -> MediaPipe pose keypoints (PoseTrack). Seams are mocked in tests."""
from dataclasses import dataclass
import numpy as np

N_LANDMARKS = 33


@dataclass
class PoseTrack:
    xy: np.ndarray        # (F, K, 2) normalized image coords [0,1], x right / y down
    visible: np.ndarray   # (F, K) visibility in [0,1]
    fps: float


def _fill_gaps(xy):
    """Linear-interpolate NaNs per landmark/coord over frames; all-NaN -> 0.5."""
    xy = np.asarray(xy, float).copy()
    F, K, C = xy.shape
    idx = np.arange(F)
    for k in range(K):
        for c in range(C):
            col = xy[:, k, c]
            valid = ~np.isnan(col)
            if valid.sum() == 0:
                xy[:, k, c] = 0.5
            elif valid.sum() < F:
                xy[:, k, c] = np.interp(idx, idx[valid], col[valid])
    return xy


def _read_frames(video_path, max_frames=None):
    """Read RGB frames + fps via imageio. Seam: monkeypatched in tests."""
    import imageio.v3 as iio
    try:
        fps = float(iio.immeta(video_path, plugin="pyav").get("fps", 30.0))
    except Exception:
        fps = 30.0
    frames = []
    for i, fr in enumerate(iio.imiter(video_path)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(np.asarray(fr)[:, :, :3])
    return frames, fps


def _detect_landmarks(frames):
    """Run MediaPipe Pose per frame -> (xy (F,K,2) with NaN gaps, visible (F,K))."""
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    F = len(frames)
    xy = np.full((F, N_LANDMARKS, 2), np.nan)
    vis = np.zeros((F, N_LANDMARKS))
    for i, fr in enumerate(frames):
        res = pose.process(np.ascontiguousarray(fr))
        if res.pose_landmarks:
            for k, lm in enumerate(res.pose_landmarks.landmark):
                xy[i, k] = [lm.x, lm.y]
                vis[i, k] = lm.visibility
    pose.close()
    return xy, vis


def estimate_poses(video_path, max_frames=None):
    frames, fps = _read_frames(video_path, max_frames)
    xy, vis = _detect_landmarks(frames)
    return PoseTrack(xy=_fill_gaps(xy), visible=np.asarray(vis, float), fps=fps)
```

- [ ] **Step 4: Run to verify they pass**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_pose.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/video/pose.py tests/test_dance_video_pose.py
git commit -m "feat(dance): MediaPipe pose -> PoseTrack (gap-filled keypoints)"
```

---

### Task 4: `scorers.py` — four pluggable saliency scorers

**Files:**
- Create: `src/kuka_sim/dance/video/scorers.py`
- Create: `tests/test_dance_video_scorers.py`

**Interfaces:**
- Consumes: `PoseTrack` (Task 3), `spectral_residual` (Task 2).
- Produces (used by Task 6):
  - `ScorerCtx` dataclass: `beat_env: np.ndarray | None = None`,
    `gray_frames: np.ndarray | None = None`.
  - `SCORERS: dict[str, Callable[[PoseTrack, ScorerCtx], np.ndarray]]` with keys
    `"energy"`, `"music"`, `"saliency"`, `"blend"`; each returns `(F,2)` focus
    points in normalized image coords.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dance_video_scorers.py`:

```python
import numpy as np
import pytest
from kuka_sim.dance.video.pose import PoseTrack
from kuka_sim.dance.video.scorers import SCORERS, ScorerCtx


def _track_two_landmarks(F=60, fps=30.0):
    # K=33; landmark 15 (a hand) whips around, everything else still.
    xy = np.full((F, 33, 2), 0.5)
    xy[:, 23] = [0.5, 0.6]; xy[:, 24] = [0.5, 0.6]   # hips (body center)
    xy[:, 11] = [0.4, 0.4]; xy[:, 12] = [0.6, 0.4]   # shoulders
    t = np.linspace(0, 4 * np.pi, F)
    xy[:, 15, 0] = 0.5 + 0.3 * np.sin(t)             # moving, extended hand
    xy[:, 15, 1] = 0.2
    return PoseTrack(xy=xy, visible=np.ones((F, 33)), fps=fps)


def test_all_four_scorers_registered():
    assert set(SCORERS) == {"energy", "music", "saliency", "blend"}


def test_energy_scorer_shape_and_range():
    focus = SCORERS["energy"](_track_two_landmarks(), ScorerCtx())
    assert focus.shape == (60, 2)
    assert focus.min() >= 0.0 and focus.max() <= 1.0


def test_energy_scorer_picks_moving_hand():
    track = _track_two_landmarks()
    focus = SCORERS["energy"](track, ScorerCtx())
    # focus should sit near the hand's y (0.2), not the still body (0.5+)
    assert np.median(focus[:, 1]) < 0.35


def test_music_scorer_requires_beat_env():
    with pytest.raises(ValueError):
        SCORERS["music"](_track_two_landmarks(), ScorerCtx())


def test_music_scorer_runs_with_beat_env():
    track = _track_two_landmarks()
    beat = np.abs(np.sin(np.linspace(0, 8 * np.pi, 60)))
    focus = SCORERS["music"](track, ScorerCtx(beat_env=beat))
    assert focus.shape == (60, 2)


def test_saliency_scorer_requires_frames():
    with pytest.raises(ValueError):
        SCORERS["saliency"](_track_two_landmarks(), ScorerCtx())


def test_blend_scorer_requires_beat_env():
    with pytest.raises(ValueError):
        SCORERS["blend"](_track_two_landmarks(), ScorerCtx())


def test_blend_scorer_runs():
    track = _track_two_landmarks()
    beat = np.abs(np.sin(np.linspace(0, 8 * np.pi, 60)))
    focus = SCORERS["blend"](track, ScorerCtx(beat_env=beat))
    assert focus.shape == (60, 2)
    assert focus.min() >= 0.0 and focus.max() <= 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_scorers.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/kuka_sim/dance/video/scorers.py`:

```python
"""Four pluggable saliency scorers: PoseTrack(+ctx) -> (F,2) focus points."""
from dataclasses import dataclass
import numpy as np
from kuka_sim.dance.video.saliency import spectral_residual

CENTER_LANDMARKS = [11, 12, 23, 24]   # shoulders + hips


@dataclass
class ScorerCtx:
    beat_env: np.ndarray | None = None      # (F,) music energy on the frame grid
    gray_frames: np.ndarray | None = None   # (F,h,w) grayscale frames


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
```

- [ ] **Step 4: Run to verify they pass**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_scorers.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/video/scorers.py tests/test_dance_video_scorers.py
git commit -m "feat(dance): four pluggable saliency scorers (energy/music/saliency/blend)"
```

---

### Task 5: `retarget.py` — focus path → `DanceTrajectory`

**Files:**
- Create: `src/kuka_sim/dance/video/retarget.py`
- Create: `tests/test_dance_video_retarget.py`

**Interfaces:**
- Consumes: `kuka_sim.dance.trajectory.{DanceTrajectory, map_to_workspace}`.
- Produces (used by Task 6): `retarget(focus_xy, fps, dt=1/120,
  box_center=(0.5,0,0.7), box_half_extents=(0.10,0.18,0.14), depth_env=None,
  video_path=None, audio_path=None) -> DanceTrajectory`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dance_video_retarget.py`:

```python
import numpy as np
from kuka_sim.dance.video.retarget import retarget


def test_retarget_center_maps_to_box_center():
    focus = np.full((30, 2), 0.5)
    traj = retarget(focus, fps=30.0, video_path="d.mp4")
    assert np.allclose(traj.ee_pos[:, :], [0.5, 0.0, 0.7], atol=1e-6)


def test_retarget_extremes_stay_in_box():
    center = np.array([0.5, 0.0, 0.7]); half = np.array([0.10, 0.18, 0.14])
    focus = np.array([[1.0, 0.0], [0.0, 1.0]] * 15)   # (30,2)
    traj = retarget(focus, fps=30.0)
    assert np.all(traj.ee_pos <= center + half + 1e-6)
    assert np.all(traj.ee_pos >= center - half - 1e-6)


def test_retarget_maps_axes():
    # image x=1 -> lateral +y (max); image y=0 -> height +z (max)
    focus = np.full((30, 2), 0.5); focus[:, 0] = 1.0; focus[:, 1] = 0.0
    traj = retarget(focus, fps=30.0)
    assert traj.ee_pos[:, 1].mean() > 0.15    # near +y edge (0 + 0.18)
    assert traj.ee_pos[:, 2].mean() > 0.82    # near +z edge (0.7 + 0.14)


def test_retarget_resamples_to_sim_grid():
    focus = np.full((30, 2), 0.5)             # 1s @30fps -> ~120 samples @1/120
    traj = retarget(focus, fps=30.0)
    assert traj.ee_pos.shape[0] == round(30 / 30.0 / (1 / 120.0))
    assert traj.dt == 1 / 120


def test_retarget_sets_pip_and_audio():
    traj = retarget(np.full((10, 2), 0.5), fps=30.0, video_path="dance.mp4")
    assert traj.pip_video_path == "dance.mp4"
    assert traj.audio_path == "dance.mp4"      # defaults to the video's own audio
```

- [ ] **Step 2: Run to verify they fail**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_retarget.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `src/kuka_sim/dance/video/retarget.py`:

```python
"""Focus path (normalized image coords) -> DanceTrajectory on the sim grid."""
import numpy as np
from kuka_sim.dance.trajectory import DanceTrajectory, map_to_workspace


def retarget(focus_xy, fps, dt=1.0 / 120.0,
             box_center=(0.5, 0.0, 0.7), box_half_extents=(0.10, 0.18, 0.14),
             depth_env=None, video_path=None, audio_path=None):
    """Map a (F,2) normalized image-space focus path to an EE trajectory.

    image-x [0,1] -> box lateral (y); (1 - image-y) -> box height (z); depth (x)
    from depth_env (in [-1,1]) or held at the box center. Resample F@fps to the
    sim grid, then map into the workspace box. pip/audio point at the source clip.
    """
    focus_xy = np.asarray(focus_xy, float)
    F = len(focus_xy)
    center = np.asarray(box_center, float)
    half = np.asarray(box_half_extents, float)

    norm = np.zeros((F, 3), float)
    norm[:, 1] = 2.0 * focus_xy[:, 0] - 1.0           # x right -> lateral y
    norm[:, 2] = 2.0 * (1.0 - focus_xy[:, 1]) - 1.0   # y down  -> height z (flip)
    if depth_env is not None:
        norm[:, 0] = np.asarray(depth_env, float)

    # resample F@fps -> N@dt
    n = int(round(F / fps / dt))
    src_t = np.arange(F) / fps
    dst_t = np.arange(n) * dt
    norm_rs = np.stack([np.interp(dst_t, src_t, norm[:, c]) for c in range(3)], axis=1)

    ee_pos = map_to_workspace(norm_rs, center, half)
    return DanceTrajectory(
        ee_pos=ee_pos, dt=dt,
        pip_video_path=video_path,
        audio_path=audio_path if audio_path is not None else video_path,
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_retarget.py -v`
Expected: all passed.

- [ ] **Step 5: Run the full video suite**

Run: `env_isaaclab/Scripts/python.exe -m pytest tests/test_dance_video_*.py -v`
Expected: all passed (pkg + saliency + pose + scorers + retarget).

- [ ] **Step 6: Commit**

```bash
git add src/kuka_sim/dance/video/retarget.py tests/test_dance_video_retarget.py
git commit -m "feat(dance): retarget focus path -> DanceTrajectory (image->box, PiP)"
```

---

### Task 6: `dance_video_demo.py` CLI + end-to-end render

**Files:**
- Create: `scripts/dance_video_demo.py`

**Interfaces:**
- Consumes: `estimate_poses`, `SCORERS`/`ScorerCtx`, `spectral_residual` (for the
  saliency ctx), `retarget`, backend `play`/`record`/`smooth_and_limit`, `launch`,
  `build_scene`, `SceneCamera`; librosa for the music/blend beat envelope.
- Produces: `out/<name>.mp4` — the arm tracking the salient point with the source
  clip inset + audio. (Smoke-verified on GPU; needs a real dance video — the one
  external input, resolved with the user at render time.)

- [ ] **Step 1: Write the demo script**

Create `scripts/dance_video_demo.py`:

```python
"""V2: drive the arm from a dance video (pose -> saliency scorer -> retarget) and
render it with the source clip inset as picture-in-picture.

Usage:
    python scripts/dance_video_demo.py --video assets/video/dance.mp4 \
        --scorer blend --out out/dance_video.mp4
    python scripts/dance_video_demo.py --video assets/video/dance.mp4 \
        --scorer energy --seconds 12
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
from kuka_sim.dance.video.pose import estimate_poses, _read_frames
from kuka_sim.dance.video.scorers import SCORERS, ScorerCtx
from kuka_sim.dance.video.saliency import _resize

USD = "assets/usd/iiwa7_r800.usd"
OUT_DIR = "out"
DT = 1.0 / 120.0
CAPTURE_EVERY = 4


def _beat_env(video_path, n_frames, fps):
    """Per-frame music energy from the video's audio (librosa), on the frame grid."""
    import librosa
    try:
        y, sr = librosa.load(video_path, sr=None, mono=True)
    except Exception:
        return np.zeros(n_frames)
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    frame_t = np.arange(n_frames) / fps
    env = np.interp(frame_t, rms_t, rms)
    lo, hi = float(env.min()), float(env.max())
    return (env - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros(n_frames)


def _gray_frames(frames):
    """Downsample + grayscale frames for the saliency scorer."""
    return np.stack([_resize(np.asarray(f).mean(axis=2), (64, 64)) for f in frames])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scorer", default="blend", choices=list(SCORERS))
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--no-pip", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    max_frames = None
    print(f"[pose] {args.video}", flush=True)
    track = estimate_poses(args.video, max_frames=max_frames)
    if args.seconds is not None:
        cut = int(args.seconds * track.fps)
        track.xy = track.xy[:cut]; track.visible = track.visible[:cut]
    F = len(track.xy)

    ctx = ScorerCtx()
    if args.scorer in ("music", "blend"):
        ctx.beat_env = _beat_env(args.video, F, track.fps)
    if args.scorer == "saliency":
        frames, _ = _read_frames(args.video, max_frames=F)
        ctx.gray_frames = _gray_frames(frames[:F])

    focus = SCORERS[args.scorer](track, ctx)
    from kuka_sim.dance.video.retarget import retarget
    pip = None if args.no_pip else args.video
    traj = retarget(focus, track.fps, dt=DT, video_path=pip, audio_path=args.video)
    traj.ee_pos = smooth_and_limit(traj.ee_pos, DT, max_speed=0.6)
    print(f"[retarget] scorer={args.scorer} {len(traj)} samples", flush=True)

    app, sim = launch(headless=True, enable_cameras=True)
    handles = build_scene(sim, USD, with_probe=False, with_surface=False)
    cam = SceneCamera(pos=(1.9, 1.9, 1.3), target=(0.5, 0.0, 0.7))
    sim.reset(); handles["arm"].initialize(); cam.initialize()

    frames_out = []

    def on_step(i):
        if i % CAPTURE_EVERY == 0:
            frames_out.append(cam.capture())

    play(sim, handles, traj, cam=cam, on_step=on_step)

    name = os.path.splitext(os.path.basename(args.video))[0]
    out = args.out or os.path.join(OUT_DIR, f"{name}_{args.scorer}.mp4")
    record(frames_out, out, fps=30, audio_path=traj.audio_path,
           pip_video_path=traj.pip_video_path)
    print(f"[OK] wrote {out} ({len(frames_out)} frames)", flush=True)
    _shutdown(app)


def _shutdown(app):
    """Force process exit within seconds of the deliverable landing (Isaac's
    app.close() can deadlock before os._exit). See dance_music_demo._shutdown."""
    import sys
    import threading
    sys.stdout.flush(); sys.stderr.flush()
    t = threading.Thread(target=app.close, daemon=True)
    t.start()
    t.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Obtain a short source dance video (external input)**

Place a short dance clip at `assets/video/dance.mp4` (the `assets/video/` dir is
gitignored like `assets/audio/`). Resolve the source with the user — a clip they
supply, or a Creative-Commons / public-domain dance clip. Add `assets/video/` to
`.gitignore` if not already covered.

- [ ] **Step 3: Smoke-render a 12s clip (controller runs this directly on GPU)**

Run: `env_isaaclab/Scripts/python.exe scripts/dance_video_demo.py --video assets/video/dance.mp4 --scorer blend --seconds 12 --out out/dance_video_smoke.mp4`
Expected: `[pose]`, `[retarget] scorer=blend …`, `[OK] wrote out/dance_video_smoke.mp4 (~90 frames)`; exit 0; no lingering python process.

- [ ] **Step 4: Verify streams + PiP**

Run: `env_isaaclab/Scripts/python.exe -c "import imageio_ffmpeg,subprocess; ff=imageio_ffmpeg.get_ffmpeg_exe(); print(subprocess.run([ff,'-i','out/dance_video_smoke.mp4'],capture_output=True).stderr.decode('utf-8','replace'))"`
Expected: a `Video: h264` + `Audio: aac` stream; the corner PiP is visible on playback.

- [ ] **Step 5: Full renders — headline + contrast scorer**

Run: `env_isaaclab/Scripts/python.exe scripts/dance_video_demo.py --video assets/video/dance.mp4 --scorer blend --out out/dance_video_blend.mp4`
Run: `env_isaaclab/Scripts/python.exe scripts/dance_video_demo.py --video assets/video/dance.mp4 --scorer energy --out out/dance_video_energy.mp4`
Expected: both print `[OK] wrote …`; both play with the arm tracking the dancer + the clip inset; the two scorers visibly differ.

- [ ] **Step 6: Commit**

```bash
git add scripts/dance_video_demo.py .gitignore
git commit -m "feat(dance): video dance demo CLI (pose -> scorer -> retarget -> render+PiP)"
```

---

## Self-Review

**Spec coverage:** pose (Task 3), saliency map (Task 2), four scorers (Task 4),
retarget (Task 5), CLI + renders (Task 6), mediapipe dep (Task 1). Spec
deliverables A–E all covered.

**Placeholder scan:** none — every code step carries full code; Task 6 Step 2 is
a real external-input step (source video), explicitly flagged in the spec.

**Type consistency:** `PoseTrack` fields (Task 3) are consumed by name in Tasks
4–6; `ScorerCtx`/`SCORERS` (Task 4) used in Task 6; `retarget` signature matches
the demo call; backend names (`DanceTrajectory`, `map_to_workspace`,
`smooth_and_limit`, `play`, `record`) match the shipped backend. Image→box axis
mapping is identical in `retarget.py` and its tests.
