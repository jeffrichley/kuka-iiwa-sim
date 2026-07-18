# Design: Arm Dance Module — V1 Music Choreography (sub-project 2 of 3)

**Date:** 2026-07-18
**Status:** Approved (design), pending implementation plan
**Author:** Jeff (with Claude Code)
**Supersedes:** `2026-07-17-arm-choreography-design.md` (that draft predates the
backend and was written in joint space; this spec targets the shipped
`DanceTrajectory` EE-space backend).

## Context

The dance module is **one backend + two front-ends** (see
`2026-07-18-dance-backend-design.md`). The shared backend is built and verified:
it takes a `DanceTrajectory` (EE poses over time, optional camera track, optional
audio/PiP), drives the arm to follow it within real limits, and records an MP4.

```
  V1 music (THIS SPEC) ──►  DanceTrajectory  ──►  BACKEND (player + recorder)  ──►  MP4 + song
```

**This spec covers only V1 (music → motion).** It produces a `DanceTrajectory`;
it does not touch the player or recorder.

## Goal

A **song-agnostic** pipeline: take any audio file, analyze it, and choreograph
the iiwa 7 R800 to it — the flange swaying, bobbing, and accenting with the
music, a camera optionally synced to musical energy, and the song muxed onto the
final MP4 by the backend.

Showcase inputs (already downloaded to `assets/audio/`): **"The Blue Danube"**
(waltz, 3/4 sway) and **"Also sprach Zarathustra"** (epic build). These are
example inputs only — drop any file in `assets/audio/` and it works.

## Non-goals

- Not real-time. The whole trajectory is precomputed from the audio timeline,
  then handed to the backend for deterministic playback.
- Not snappy/breakdance. The arm has real velocity limits; the backend's
  `smooth_and_limit` already clamps EE speed, so choreography reads as graceful
  waltz/tai-chi, not violent. V1 authors *desired* motion and lets the backend
  make it physically playable.
- No inverse kinematics or joint-space authoring. Motion is authored as
  **EE-space offsets in the workspace box** (what the backend consumes).
- No new heavy deps beyond `librosa` (added under an optional `[dance]` extra).

## Architecture — `src/kuka_sim/dance/music/`

```
  features.py        librosa analysis -> FeatureTrack resampled to sim dt
  moves.py           named EE-space gesture generators (pure numpy)
  choreographer.py   FeatureTrack -> DanceTrajectory (ee_pos/ee_quat + camera)
```

Plus `scripts/dance_music_demo.py` (CLI) and tests under `tests/dance/music/`.

### `features.py` — audio → FeatureTrack (pure logic, unit-tested with a stub)

```python
@dataclass
class FeatureTrack:
    dt: float                 # seconds per sample (== sim dt, 1/120)
    n: int                    # number of samples
    rms: np.ndarray           # (n,) normalized energy envelope in [0,1]
    bass: np.ndarray          # (n,) normalized low-band energy in [0,1]
    mid: np.ndarray           # (n,) normalized mid-band energy in [0,1]
    treble: np.ndarray        # (n,) normalized high-band energy in [0,1]
    beats: np.ndarray         # (n,) 0/1 impulse train, 1 on a beat sample
    onsets: np.ndarray        # (n,) 0/1 impulse train, 1 on an onset sample
    tempo: float              # estimated BPM

def analyze(audio_path: str, dt: float = 1/120) -> FeatureTrack:
    """librosa: load -> beat_track, rms, mel-band split (bass/mid/treble),
    onset_detect. Resample every frame-rate series to the sim grid (n = ceil(
    duration/dt)) so sample i corresponds to song time i*dt. Each band/rms is
    min-max normalized to [0,1] over the track."""
```

- Band split: mel spectrogram, sum mel bins into three contiguous ranges
  (bass / mid / treble), then normalize each.
- Beats/onsets: convert `librosa` frame indices → nearest sim-sample index →
  set 1 in an otherwise-zero (n,) array.
- **Testability:** `analyze` calls one thin internal seam,
  `_load_audio(path) -> (y, sr)`. Tests monkeypatch that seam to feed a
  synthesized signal (e.g. a click track), so the librosa math runs on known
  input without shipping an audio fixture. Resampling/normalization/impulse
  logic is factored into small pure helpers tested directly.

### `moves.py` — EE-space gesture generators (pure numpy, unit-tested)

Each move is a function `(t: np.ndarray, **params) -> np.ndarray` returning an
`(n, 3)` **offset** (metres) in workspace-box-local axes (x=depth, y=lateral,
z=height). Offsets are summed by the choreographer, then mapped into the box and
speed-limited by the backend. Starting library (~6):

- `sway(t, freq, amp)` — lateral (y) sine; the waltz base motion.
- `bob(t, freq, amp)` — vertical (z) sine, typically 2× sway freq.
- `reach(t, env, amp)` — depth (x) push driven by an envelope (e.g. RMS).
- `circle(t, freq, amp)` — y/z circle for flourishes.
- `figure_eight(t, freq, amp)` — Lissajous y/z.
- `accent(impulse, decay, amp, axis)` — exponentially-decaying kick on each
  impulse sample (beats/onsets), a transient pop on the given axis.

All pure `numpy`; no Isaac. Amplitudes are small (a few cm) so the summed motion
stays inside the dexterous box.

### `choreographer.py` — FeatureTrack → DanceTrajectory (pure logic, unit-tested)

```python
def choreograph(track: FeatureTrack, style: str = "waltz",
                camera: str = "hero",
                box_center=(0.5, 0.0, 0.7),
                box_half_extents=(0.10, 0.18, 0.14)) -> DanceTrajectory:
    """Combine moves, modulated by the FeatureTrack, into EE positions; add an
    orientation flourish and an optional camera track. Returns a DanceTrajectory
    (ee_pos already mapped into the box; NOT yet speed-limited — the demo script
    calls smooth_and_limit, matching the backend contract)."""
```

Mapping (the "choreography"), for the default `waltz` style:
- **bass → base sway:** `sway` amplitude scales with per-sample `bass`.
- **rms → depth reach + overall gain:** `reach` uses the RMS envelope; global
  amplitude scales with RMS so quiet passages are small, swells are large.
- **mid → bob:** vertical `bob` amplitude scales with `mid`.
- **beats → accent:** an `accent` kick on each beat sample (the visible "pulse").
- **treble/onsets → wrist flourish:** builds `ee_quat` as a small roll/yaw
  oscillation whose amplitude tracks `treble`, with an onset-triggered accent.
  Base orientation is `FORWARD_QUAT`; the flourish is a bounded perturbation.

`style` selects a parameter preset (freqs/amps/which bands drive which axis).
Ship two: `"waltz"` (gentle 3/4, for Blue Danube) and `"epic"` (slower, larger
reach and a stronger RMS gain, for Zarathustra). Unknown style → `ValueError`.

Camera:
- `"hero"` → `cam_pos=None` (backend uses its fixed 3/4 hero shot).
- `"cinematic"` → an `(n,3)` slow orbit around the box center + an RMS-driven
  radial push-in on peaks; `cam_target` fixed on the box center. Restrained so it
  never fights the arm.

Orientation is built as proper unit quaternions (small-angle roll/yaw composed
with `FORWARD_QUAT`), returned as `ee_quat` `(n,4)` wxyz.

### `scripts/dance_music_demo.py` — CLI (sim-runtime, smoke-verified)

```
python scripts/dance_music_demo.py --song assets/audio/blue_danube.mp3 \
       --style waltz --camera hero --out out/blue_danube.mp4
```

1. `track = analyze(song, dt=1/120)`
2. `traj = choreograph(track, style, camera)`
3. `traj.ee_pos = smooth_and_limit(traj.ee_pos, dt, max_speed=0.6)`  (backend helper)
4. set `traj.audio_path = song`
5. `launch` → `build_scene(with_probe=False, with_surface=False)` (bare arm) →
   `play(sim, handles, traj, cam=SceneCamera(...), on_step=capture)`
6. `record(frames, out, fps=30, audio_path=traj.audio_path)` → MP4 with the song
7. `os._exit(0)` (Isaac cleanup, per the established pattern)

Optional `--seconds N` to render only the first N seconds (fast iteration /
smoke). Default renders the whole song.

## Data flow

```
  song.mp3 ─analyze─► FeatureTrack ─choreograph─► DanceTrajectory
                                                      │ smooth_and_limit
                                                      ▼
                                        BACKEND play + record ─► out/song.mp4 (+audio)
```

## Deliverables (each independently verifiable)

- **A — Features.** `analyze` + helpers produce a `FeatureTrack` on the sim grid.
  *Verify:* pytest with a monkeypatched `_load_audio` feeding a synthetic click
  track — beats land on the right samples, bands normalize to [0,1], `n` matches
  `ceil(duration/dt)`, impulse trains are 0/1.
- **B — Moves.** The six generators return correct-shape offsets with expected
  structure. *Verify:* pytest — `sway` is lateral-only and bounded by `amp`;
  `accent` decays from each impulse; shapes are `(n,3)`.
- **C — Choreographer.** `choreograph` returns a valid `DanceTrajectory`:
  `ee_pos` inside the box, `ee_quat` unit-norm, camera track present/absent per
  mode, unknown style raises. *Verify:* pytest on a small synthetic FeatureTrack.
- **D — End-to-end render.** `dance_music_demo.py` renders **Blue Danube**
  (`waltz`/`hero`) and **Zarathustra** (`epic`/`cinematic`) to MP4s with audio.
  *Verify (smoke, on GPU):* the arm sways/accents in time with the music; the
  output has a video + AAC audio stream (ffprobe); a short `--seconds 12` clip is
  the fast check before the full render.

## Testing strategy

- **Pure logic → pytest/TDD** (subagent-implementable): `features.py` helpers +
  `analyze` (stubbed load), all of `moves.py`, all of `choreographer.py`.
- **Sim-runtime → smoke** (run directly on GPU, not via subagents, per the
  module's Isaac-coupling convention): the demo render.

## Dependencies

- `librosa` — add to an optional `[dance]` extra in `pyproject.toml` (keeps the
  core install lean). Its transitive deps (numba, soundfile, audioread) come with
  it. ffmpeg is already present via `imageio-ffmpeg`.

## Open questions (non-blocking, defer to build)

- **Move-library richness.** Start with the six above; add more only if the
  motion reads thin on the showcase songs.
- **Style count.** Ship `waltz` + `epic`. A third preset is trivial to add later.
- **Beat-locked vs. continuous.** Default blends continuous band-driven motion
  with beat/onset accents. If accents dominate or wash out, tune amplitudes at
  smoke time — a parameter change, not a redesign.
