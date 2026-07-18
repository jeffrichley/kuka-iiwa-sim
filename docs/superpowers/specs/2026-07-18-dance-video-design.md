# Design: Arm Dance Module — V2 Video → Arm (sub-project 3 of 3)

**Date:** 2026-07-18
**Status:** Approved (design), pending implementation plan
**Author:** Jeff (with Claude Code)

## Context

The dance module is **one backend + two front-ends** (see
`2026-07-18-dance-backend-design.md`). Backend and V1 (music) are done. V2 turns
a **dance video** into arm motion:

```
  dance.mp4 ─pose─► keypoints ─scorer─► focus point/frame ─retarget─►
                                                    DanceTrajectory ─► BACKEND ─► MP4 (+ PiP of the source video)
```

**This spec covers only V2.** It produces a `DanceTrajectory`; it does not touch
the player or recorder. The source video rides along as picture-in-picture via
the backend's existing `pip_video_path`, so the arm and the dancer play in sync.

## Goal

Take any dance video, estimate the dancer's pose, decide **which moving point is
"most interesting" at each moment** (the saliency question — it changes through
the dance), have the arm's flange track that point, and render the arm dancing
with the original clip inset in the corner.

Per the user's direction, ship **all four** saliency scorers as pluggable,
selectable strategies.

## Non-goals

- Not full-body retargeting (7-DOF arm ≠ human skeleton). We track **one focus
  point** — the most interesting moving location — not every limb. This is the
  approved framing ("follow the most interesting moving point").
- Not real-time. Whole video analyzed offline → precomputed trajectory → played.
- No exact IK. The impedance backend follows the focus path with graceful lag.
- Pose estimator choice is settled: **MediaPipe Pose** ("any estimator — the
  easiest good one"). No OpenPose (heavier install, no Windows wheels).

## Architecture — `src/kuka_sim/dance/video/`

```
  pose.py        MediaPipe pose: video -> keypoints (F,33,2) + fps  (thin seam)
  saliency.py    pure-numpy spectral-residual saliency map (for scorer #3)
  scorers.py     4 pluggable scorers: keypoints(+ctx) -> focus point per frame
  retarget.py    focus path -> smoothed -> workspace box -> DanceTrajectory
```

Plus `scripts/dance_video_demo.py` (CLI) and `tests/test_dance_video_*.py`.

### `pose.py` — video → keypoints (seam for tests)

```python
@dataclass
class PoseTrack:
    xy: np.ndarray        # (F, K, 2) landmark image coords, normalized [0,1] (x right, y down)
    visible: np.ndarray   # (F, K) visibility/confidence in [0,1]
    fps: float

def estimate_poses(video_path: str, max_frames: int | None = None) -> PoseTrack:
    """Read frames (imageio) and run MediaPipe Pose (K=33 landmarks). Missing
    detections -> previous frame's coords with visibility 0."""
```

- Testability: `estimate_poses` calls two seams — `_read_frames(path) ->
  (frames, fps)` and `_pose_landmarker()` (the MediaPipe model). Tests
  monkeypatch both, so the assembly logic (gap-fill, normalization, stacking) is
  unit-tested without MediaPipe or a video file.
- MediaPipe is added under a `[dance]` optional dep. Frames via `imageio`
  (already present through `imageio-ffmpeg`).

### `saliency.py` — spectral-residual saliency (pure numpy, unit-tested)

```python
def spectral_residual(gray: np.ndarray, out_hw=(64, 64)) -> np.ndarray:
    """Hou & Zhang spectral-residual saliency: resize to out_hw, FFT, subtract a
    smoothed log-amplitude spectrum, inverse-FFT, square, blur, normalize to
    [0,1]. Returns an (H,W) saliency map. No deep model — ~20 lines of numpy."""
```

This is the dependency-light realization of scorer #3: it estimates "where the
eye goes" from image structure, testable on synthetic frames (a bright blob →
peak saliency at the blob).

### `scorers.py` — the four strategies (pure logic, unit-tested)

Each scorer is `score(track: PoseTrack, ctx: ScorerCtx) -> np.ndarray` returning
`(F, 2)` **focus points** in normalized image coords — the point the arm tracks
each frame. Registered in `SCORERS: dict[str, Callable]` keyed
`"energy" | "music" | "saliency" | "blend"`.

`ScorerCtx` carries optional inputs: `beat_env: np.ndarray | None` (per-frame
music energy on the video's frame grid, from librosa — for music/blend),
`gray_frames: np.ndarray | None` (downsampled grayscale frames — for saliency).

- **`energy` (motion energy = speed × reach):** per landmark per frame, `speed`
  = frame-to-frame displacement, `reach` = distance from the body center (mean of
  hips/shoulders). Score = `speed * reach`, smoothed over ~1 s. Winner landmark's
  xy is the focus. Dependency-free, robust.
- **`music` (musical correlation):** for each landmark, correlate its speed
  series against `beat_env` in a sliding window; the most on-beat landmark wins
  per frame. Requires `beat_env`.
- **`saliency` (visual saliency):** for each frame, `spectral_residual` → map;
  the landmark sitting on the hottest map cell wins. Requires `gray_frames`.
- **`blend`:** normalize the per-landmark energy and music scores, combine
  `0.6*energy + 0.4*music`, winner wins. Requires `beat_env`.

Winner selection is smoothed (hysteresis / short-window argmax) so the focus
doesn't jitter between landmarks frame to frame.

### `retarget.py` — focus path → DanceTrajectory (pure logic, unit-tested)

```python
def retarget(focus_xy, fps, dt=1/120,
             box_center=(0.5,0,0.7), box_half_extents=(0.10,0.18,0.14),
             depth_env=None, video_path=None, audio_path=None) -> DanceTrajectory:
    """Map a normalized image-space focus path (F,2) to an EE trajectory on the
    sim grid. image-x -> box lateral(y); (1-image-y) -> box height(z); depth(x)
    from depth_env or held at box center. Resample F@fps -> N@dt, center to [-1,1],
    map_to_workspace, and set pip_video_path/audio_path so the backend insets the
    source clip and plays its audio."""
```

- Coordinate mapping: image x∈[0,1] (right) → lateral **y**; image y∈[0,1]
  (down) → height **z** via `(1 - y)` (screen-up = arm-up); depth **x** constant
  (or driven by an energy envelope for a subtle push on big moments).
- Resampling: `focus_xy` is at video `fps`; resample each channel onto the sim
  grid (`N = round(F/fps/dt)`) using the same interp helper family as V1.
- Orientation: hold `FORWARD_QUAT` (V2 tracks a *point*, not a limb orientation).
- `pip_video_path = video_path`, `audio_path = audio_path or video_path` (the
  backend pulls the video's own audio stream).

### `scripts/dance_video_demo.py` — CLI (sim-runtime, smoke-verified on GPU)

```
python scripts/dance_video_demo.py --video assets/video/dance.mp4 \
    --scorer blend --out out/dance_video.mp4
```

1. `track = estimate_poses(video)`
2. build `ScorerCtx` (beat_env from librosa on the video's audio if the scorer
   needs it; gray_frames if saliency)
3. `focus = SCORERS[scorer](track, ctx)`
4. `traj = retarget(focus, track.fps, video_path=video)`
5. `traj.ee_pos = smooth_and_limit(traj.ee_pos, dt, max_speed=0.6)`
6. backend `play` → capture → `record(frames, out, audio_path=traj.audio_path,
   pip_video_path=traj.pip_video_path)`
7. robust `_shutdown(app)` (daemon-thread close + `os._exit(0)`), per
   `dance_music_demo.py`.

`--seconds N` renders only the first N seconds (fast smoke). `--no-pip` drops the
inset.

## Data flow

```
  dance.mp4 ─estimate_poses─► PoseTrack ─┐
                                          ├─ SCORERS[scorer] ─► focus_xy (F,2)
  (audio→librosa beat_env, frames→gray) ─┘        │ retarget
                                                   ▼
                              DanceTrajectory(pip=dance.mp4) ─► backend ─► out.mp4
```

## Deliverables (each independently verifiable)

- **A — Pose.** `estimate_poses` assembles a `PoseTrack` (gap-filled, normalized).
  *Verify:* pytest with both seams monkeypatched (synthetic landmark stream) —
  shape `(F,33,2)`, gaps carried forward with visibility 0, fps passed through.
- **B — Saliency.** `spectral_residual` peaks on a bright blob and normalizes to
  [0,1]. *Verify:* pytest on synthetic frames.
- **C — Scorers.** All four registered; each returns `(F,2)` focus in [0,1]; on a
  crafted PoseTrack the energy scorer picks the fastest-and-most-extended
  landmark; music/blend/saliency pick the intended landmark given a crafted ctx;
  missing required ctx raises. *Verify:* pytest.
- **D — Retarget.** Focus path maps inside the box, resamples to the sim grid
  (`len == round(F/fps/dt)`), sets `pip_video_path`/`audio_path`. *Verify:* pytest.
- **E — End-to-end render.** `dance_video_demo.py` on a real short dance clip
  produces `out/dance_video.mp4`: the arm tracks the salient point with the clip
  inset and audio. *Verify (smoke, GPU):* a `--seconds 12` clip has video + audio
  streams and a visible corner PiP; then a full render for the chosen scorer(s).

## Testing strategy

- **Pure logic → pytest/TDD** (subagent-implementable): `saliency.py`,
  `scorers.py`, `retarget.py`, and `estimate_poses` assembly (seams mocked).
- **Sim-runtime + MediaPipe → smoke** (run directly on GPU, not via subagents):
  the render, which needs a real video + the MediaPipe model.

## Dependencies

- `mediapipe` — add to the `[dance]` extra (pose estimation). Ships Windows/
  Py3.11 wheels.
- `imageio` (already via `imageio-ffmpeg`) for frame reading.
- `librosa` (already in `[dance]`) for the music/blend scorers.
- Saliency is pure numpy — **no** opencv-contrib / deep saliency model.

## Open questions / decisions deferred to build

- **Source video.** The one external input V2 needs for the render deliverable.
  Resolve at the render step: use a short Creative-Commons / public-domain dance
  clip, or a clip the user supplies. All code + unit tests are built and green
  before this is needed.
- **Focus smoothing strength.** Winner-hysteresis window (~0.5–1 s) tuned at
  smoke time so the arm neither jitters nor lags the dancer.
- **Depth motion.** Default holds depth constant; optionally drive a small `x`
  push from motion energy if the result reads flat. Parameter, not redesign.
- **Which scorer(s) to render.** All four are built; render `blend` as the
  headline plus at least one contrast (`energy`) to show the difference.
