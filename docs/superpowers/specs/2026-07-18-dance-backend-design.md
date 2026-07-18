# Design: Arm Dance Module — Shared Backend (sub-project 1 of 3)

**Date:** 2026-07-18
**Status:** Approved (design), pending implementation plan
**Author:** Jeff (with Claude Code)

## The module (context)

A bonus module that makes the simulated iiwa **dance**, in two flavors that share
one engine:

- **V1 — Music choreography:** a song → procedural motion.
- **V2 — Video → arm:** a dance video → pose estimation → **saliency-driven
  retargeting** → the arm follows the "most interesting" moving point.

Both reduce to the same thing: produce a **time-series of arm targets**, then
play it in the sim and film it. So the module is **one backend + two
front-ends**, built as three sub-projects (backend → V1 → V2), each with its own
spec → plan → build.

```
  V1 music  ─┐
             ├─►  DanceTrajectory  ──►  BACKEND (player + recorder)  ──►  MP4
  V2 video  ─┘        (this spec's contract)
```

**This spec covers only the shared backend.** V1 and V2 get their own specs.

## Goal (backend)

Take a `DanceTrajectory` (end-effector poses over time, optionally a camera
track), drive the arm to follow it in the sim within the arm's real limits, and
record it to a video — with optional **audio** (the song) and **picture-in-picture**
(the source dance video, time-synced in a corner).

## Non-goals

- Generating motion — that's V1/V2. The backend only *plays and films* a
  trajectory handed to it.
- Exact trajectory tracking / inverse kinematics — the impedance controller
  follows EE-pose targets with a little graceful lag, which suits dance. (IK is a
  possible later upgrade, out of scope here.)
- New heavy dependencies. The backend uses only what's already installed;
  `librosa`/`mediapipe` belong to V1/V2.

## The contract: `DanceTrajectory`

The single interface both front-ends produce and the backend consumes.

```python
@dataclass
class DanceTrajectory:
    ee_pos: np.ndarray            # (N, 3) target flange positions, world frame
    ee_quat: np.ndarray | None    # (N, 4) target orientations (wxyz); None -> hold FORWARD_QUAT
    dt: float                     # seconds between samples (== sim dt, 1/120)
    cam_pos: np.ndarray | None = None      # (N, 3) or None -> a fixed hero camera
    cam_target: np.ndarray | None = None   # (N, 3)
    audio_path: str | None = None          # song to mux onto the video
    pip_video_path: str | None = None      # source dance video for picture-in-picture
```

`N = duration * 120`. A 30 s clip is 3600 samples — fine.

## Architecture — `src/kuka_sim/dance/`

```
  trajectory.py   DanceTrajectory dataclass + helpers (below)
  player.py       play(sim, handles, traj, gains, on_step) -> drives the arm
  recorder.py     encode frames -> MP4, mux audio, composite PiP (ffmpeg)
```

### `trajectory.py` — pure-logic helpers (unit-tested, no Isaac)

- `smooth_and_limit(pos, dt, max_speed, smooth_win) -> pos` — low-pass smooth the
  path and clamp per-sample EE speed to `max_speed` (m/s). A human dancer moves
  faster than the arm should; the front-ends hand over "desired" motion and the
  backend makes it physically graceful. **This is why the arm looks elegant, not
  violent.**
- `map_to_workspace(norm_pos, box) -> pos` — map a normalized motion (e.g. a
  salient point's path in [-1,1], or music-driven offsets) into the arm's
  dexterous workspace box (a center + half-extents in front of the arm, ~0.4–0.6 m
  out, mid-height). Used by V1/V2 to place motion where the arm has authority.

### `player.py` — sim-runtime (smoke-verified)

```python
def play(sim, handles, traj, stiffness=..., damping=..., on_step=None):
    """Drive the arm to follow traj.ee_pos/ee_quat via CartesianImpedanceController.
    Calls on_step(i) each tick (the recorder grabs a frame there)."""
```

- Reuses `CartesianImpedanceController` (high stiffness for tracking). Each tick:
  set the pose target to `traj.ee_pos[i]` (+ quat or `FORWARD_QUAT`), `ctrl.apply()`,
  `arm.write()`, `sim.step()`, `arm.update()`, `on_step(i)`.
- If `cam_pos` is present, moves the camera each tick (a choreographed camera);
  else a fixed hero camera.
- Torque is already clamped by the controller; speed is limited by
  `smooth_and_limit` upstream — so the arm can't be commanded past its limits.

### `recorder.py` — video assembly (ffmpeg via `imageio-ffmpeg`)

```python
def record(frames, out_path, fps=30, audio_path=None, pip_video_path=None):
    """Encode frames to MP4; optionally mux audio and overlay a corner PiP."""
```

- Base video: `frames_to_mp4` (existing).
- **Audio + PiP in one ffmpeg pass** using `imageio_ffmpeg.get_ffmpeg_exe()`:
  - audio only: `-i base -i song -c:v copy -c:a aac -shortest out`
  - PiP: `-filter_complex "[1:v]scale=iw/4:-1[p];[0:v][p]overlay=W-w-20:H-h-20"`
  - both: combine the overlay filter with the audio map.
- The ffmpeg **command is built by a pure function** `build_ffmpeg_cmd(...)` so it
  can be unit-tested without running ffmpeg; a separate integration test runs it on
  tiny synthetic inputs.
- Because the render is time-synced (frame N = dance time N·dt), the PiP dancer
  lines up with the arm automatically.

## Data flow

1. A front-end builds a `DanceTrajectory` (raw desired motion).
2. `smooth_and_limit` makes the EE path physically playable.
3. `play` drives the arm to follow it; a per-step callback captures camera frames.
4. `record` encodes the frames, muxes the song, and overlays the source PiP.

## Deliverables (each independently verifiable)

- **A — Trajectory helpers.** `smooth_and_limit`, `map_to_workspace`, and
  `build_ffmpeg_cmd` implemented and **unit-tested** (pure logic, no Isaac).
  *Verify:* pytest — a too-fast path is clamped to `max_speed`; a normalized path
  maps into the box; the ffmpeg command has the right filter/map for each of
  {plain, audio, pip, audio+pip}.
- **B — Player.** `play` drives the arm along a canned **circle** EE trajectory.
  *Verify (smoke):* the flange traces a smooth circle in the workspace; no limit
  violations; clean run.
- **C — Recorder.** `record` produces an MP4 from captured frames; audio + PiP
  variants produce a muxed/overlaid MP4. *Verify:* run on a short canned clip →
  `out/dance_backend_demo.mp4` plays; the audio/PiP variants contain the extra
  streams (ffprobe or visual check).
- **D — End-to-end smoke.** A `scripts/dance_backend_demo.py` plays a canned
  trajectory with a camera track and records it (optionally with a stock
  audio/PiP) — proving the backend before any front-end exists.

## Testing strategy

- **Pure logic → pytest:** `smooth_and_limit`, `map_to_workspace`,
  `build_ffmpeg_cmd`. (The FakeArm pattern isn't needed — these are array/string
  functions.)
- **Sim-runtime → smoke:** `play` (needs the arm + GPU), verified by the canned
  circle. Consistent with how the rest of the sim layer is validated.

## Dependencies

None new for the backend (ffmpeg ships via `imageio-ffmpeg`). An optional
`[dance]` extra in `pyproject.toml` will be added by V1 (`librosa`) and V2
(`mediapipe`) — not here.

## Open questions (non-blocking)

- **Orientation along the path.** Default: hold `FORWARD_QUAT` (flange forward,
  like the force demo) so the tool "points" as it moves. A later option: orient
  the flange along the velocity direction for a more expressive look. Default is
  fine for the backend.
- **Tracking lag.** Impedance tracking lags a moving target slightly (graceful
  follow-through). If a front-end needs tight sync, raise stiffness or add IK —
  revisit per front-end, not here.
- **Camera default.** A fixed 3/4 hero shot unless the trajectory carries a
  camera track (V1/V2 may choreograph the camera).
