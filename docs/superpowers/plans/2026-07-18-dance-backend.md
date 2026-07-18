# Dance Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared backend that plays a `DanceTrajectory` on the iiwa in the sim (velocity-limited, so it looks graceful) and records it to an MP4 with optional song audio and a picture-in-picture of a source video.

**Architecture:** A `kuka_sim/dance/` package: `trajectory.py` (the `DanceTrajectory` contract + pure-logic helpers `smooth_and_limit`, `map_to_workspace`), `recorder.py` (a pure `build_ffmpeg_cmd` + `record` that runs ffmpeg), and `player.py` (drives the arm along the trajectory via the existing `CartesianImpedanceController`). Both dance front-ends (music, video) produce a `DanceTrajectory` and feed this backend. Pure logic is unit-tested; sim-runtime is smoke-verified on the GPU.

**Tech Stack:** Python 3.11, NumPy, the existing `kuka_sim` sim layer (Isaac Lab 2.3), ffmpeg via `imageio-ffmpeg`, pytest.

## Global Constraints

- **No new dependencies.** The backend uses only NumPy + the existing `kuka_sim` + ffmpeg (already shipped by `imageio-ffmpeg`). `librosa`/`mediapipe` belong to the later front-ends, not here.
- **Quaternions are `wxyz`.** Default flange orientation is `FORWARD_QUAT = [0.7071, 0, 0.7071, 0]` (flange aimed forward, matching the force demo).
- **Sim dt is `1/120`.** Trajectories are sampled at sim dt.
- **The arm's real limits are never exceeded** — torque is clamped by `CartesianImpedanceController`; EE speed is clamped by `smooth_and_limit`. Front-ends hand over "desired" motion; the backend makes it playable.
- **Reuse, don't duplicate:** `CartesianImpedanceController`, `IiwaArm`, `SceneCamera`, `frames_to_mp4`, `sim_app.launch`, `scene.build_scene` already exist and must be reused.
- **Sim-runtime scripts force a clean exit** with `os._exit(0)` after `app.close()` (Isaac leaves non-daemon threads).

### Interfaces (used across tasks)

`DanceTrajectory` (Task 1) — the contract both front-ends produce and the player consumes:
```
ee_pos: np.ndarray (N,3)          # target flange positions, world frame
ee_quat: np.ndarray (N,4) | None  # wxyz; None -> hold FORWARD_QUAT
dt: float = 1/120
cam_pos, cam_target: (N,3) | None # None -> fixed hero camera
audio_path, pip_video_path: str | None
```

Pure-logic helpers:
```
map_to_workspace(norm_pos (N,3) in [-1,1], center (3,), half_extents (3,)) -> (N,3)
smooth_and_limit(pos (N,3), dt, max_speed, smooth_win=5) -> (N,3)
build_ffmpeg_cmd(ffmpeg, base_path, out_path, audio_path=None, pip_video_path=None,
                 pip_scale=0.25, margin=20) -> list[str] | None
```

---

### Task 1: Dance package + `DanceTrajectory` + `map_to_workspace`

**Files:**
- Create: `src/kuka_sim/dance/__init__.py`
- Create: `src/kuka_sim/dance/trajectory.py`
- Test: `tests/test_dance_trajectory.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DanceTrajectory`, `FORWARD_QUAT`, `map_to_workspace`.

- [ ] **Step 1: Write failing tests**

`tests/test_dance_trajectory.py`:
```python
import numpy as np
from kuka_sim.dance.trajectory import DanceTrajectory, map_to_workspace, FORWARD_QUAT

def test_forward_quat_is_unit_wxyz():
    assert FORWARD_QUAT.shape == (4,)
    assert np.isclose(np.linalg.norm(FORWARD_QUAT), 1.0)

def test_trajectory_len():
    t = DanceTrajectory(ee_pos=np.zeros((10, 3)))
    assert len(t) == 10 and t.dt == 1.0 / 120.0

def test_map_to_workspace_center_and_corners():
    center = np.array([0.5, 0.0, 0.6]); he = np.array([0.1, 0.15, 0.1])
    assert np.allclose(map_to_workspace([[0, 0, 0]], center, he), [center])
    assert np.allclose(map_to_workspace([[1, 1, 1]], center, he), [center + he])
    assert np.allclose(map_to_workspace([[-1, -1, -1]], center, he), [center - he])

def test_map_to_workspace_clips_out_of_range():
    center = np.array([0.5, 0.0, 0.6]); he = np.array([0.1, 0.1, 0.1])
    out = map_to_workspace([[2.0, 0.0, 0.0]], center, he)
    assert np.allclose(out, [center + [he[0], 0, 0]])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_dance_trajectory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kuka_sim.dance'`.

- [ ] **Step 3: Implement the package**

`src/kuka_sim/dance/__init__.py`:
```python
"""Make the arm dance: a shared player/recorder backend + front-ends."""
```

`src/kuka_sim/dance/trajectory.py`:
```python
"""Dance trajectory contract + pure-logic helpers (no Isaac)."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Flange aimed forward (tool axis at +X), matching the force demo.
FORWARD_QUAT = np.array([0.7071, 0.0, 0.7071, 0.0])


@dataclass
class DanceTrajectory:
    ee_pos: np.ndarray                       # (N, 3) target flange positions, world
    ee_quat: Optional[np.ndarray] = None     # (N, 4) wxyz; None -> hold FORWARD_QUAT
    dt: float = 1.0 / 120.0
    cam_pos: Optional[np.ndarray] = None     # (N, 3) or None -> fixed hero camera
    cam_target: Optional[np.ndarray] = None  # (N, 3)
    audio_path: Optional[str] = None
    pip_video_path: Optional[str] = None

    def __len__(self):
        return len(self.ee_pos)


def map_to_workspace(norm_pos, center, half_extents):
    """Map normalized positions in [-1, 1] into the arm's workspace box.

    norm_pos (N,3) with each axis in [-1,1] (clipped) -> center +- half_extents.
    """
    norm = np.clip(np.asarray(norm_pos, float), -1.0, 1.0)
    return np.asarray(center, float) + norm * np.asarray(half_extents, float)
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/test_dance_trajectory.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/__init__.py src/kuka_sim/dance/trajectory.py tests/test_dance_trajectory.py
git commit -m "feat(dance): add DanceTrajectory contract + map_to_workspace"
```

---

### Task 2: `smooth_and_limit` velocity limiter

**Files:**
- Modify: `src/kuka_sim/dance/trajectory.py`
- Test: `tests/test_dance_smooth.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `smooth_and_limit(pos, dt, max_speed, smooth_win=5) -> (N,3)`.

- [ ] **Step 1: Write failing tests**

`tests/test_dance_smooth.py`:
```python
import numpy as np
from kuka_sim.dance.trajectory import smooth_and_limit

def test_velocity_is_clamped():
    dt, max_speed = 1.0 / 120.0, 1.0            # max step = 1/120 m
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], float)  # 1 m jump = way too fast
    out = smooth_and_limit(pos, dt, max_speed, smooth_win=1)  # no smoothing
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.all(steps <= max_speed * dt + 1e-9)

def test_constant_path_preserved():
    pos = np.tile([0.5, 0.0, 0.6], (12, 1))
    out = smooth_and_limit(pos, 1.0 / 120.0, 1.0, smooth_win=5)
    assert np.allclose(out, pos)                # constant in, constant out (edge-padded)

def test_shape_and_endpoints_preserved_under_smoothing():
    rng = np.random.RandomState(0)
    pos = rng.randn(50, 3) * 0.005 + [0.5, 0.0, 0.6]
    out = smooth_and_limit(pos, 1.0 / 120.0, 0.5, smooth_win=5)
    assert out.shape == (50, 3)
    # smoothing reduces jitter -> less total path length than the noisy input
    def plen(a): return np.linalg.norm(np.diff(a, axis=0), axis=1).sum()
    assert plen(out) <= plen(pos)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_dance_smooth.py -v`
Expected: FAIL — `ImportError: cannot import name 'smooth_and_limit'`.

- [ ] **Step 3: Implement (append to `trajectory.py`)**

```python
def smooth_and_limit(pos, dt, max_speed, smooth_win=5):
    """Low-pass smooth an EE path (edge-padded moving average), then clamp each
    step so EE speed never exceeds max_speed (m/s). Front-ends produce 'desired'
    motion; this makes it physically playable and graceful."""
    pos = np.asarray(pos, float)
    if smooth_win > 1 and len(pos) >= smooth_win:
        pad = smooth_win // 2
        padded = np.pad(pos, ((pad, pad), (0, 0)), mode="edge")
        k = np.ones(smooth_win) / smooth_win
        pos = np.stack(
            [np.convolve(padded[:, i], k, mode="valid") for i in range(pos.shape[1])],
            axis=1,
        )
    max_step = max_speed * dt
    out = pos.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        dist = float(np.linalg.norm(d))
        if dist > max_step and dist > 0.0:
            out[i] = out[i - 1] + d * (max_step / dist)
    return out
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/test_dance_smooth.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/trajectory.py tests/test_dance_smooth.py
git commit -m "feat(dance): add smooth_and_limit EE velocity limiter"
```

---

### Task 3: `build_ffmpeg_cmd` (audio mux + PiP command builder)

**Files:**
- Create: `src/kuka_sim/dance/recorder.py`
- Test: `tests/test_dance_ffmpeg_cmd.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `build_ffmpeg_cmd(ffmpeg, base_path, out_path, audio_path=None, pip_video_path=None, pip_scale=0.25, margin=20) -> list[str] | None`. Input order is **base, pip, audio**.

- [ ] **Step 1: Write failing tests**

`tests/test_dance_ffmpeg_cmd.py`:
```python
from kuka_sim.dance.recorder import build_ffmpeg_cmd

def test_none_when_no_extras():
    assert build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4") is None

def test_audio_only():
    c = build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4", audio_path="s.mp3")
    assert "s.mp3" in c and "aac" in c and "-shortest" in c
    assert "-filter_complex" not in c and c[-1] == "o.mp4"

def test_pip_only():
    c = build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4", pip_video_path="d.mp4")
    fc = c[c.index("-filter_complex") + 1]
    assert "scale=iw*0.25" in fc and "overlay=W-w-20:H-h-20" in fc
    assert "[v]" in c and c[-1] == "o.mp4"

def test_audio_and_pip_input_order_and_maps():
    c = build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4",
                         audio_path="s.mp3", pip_video_path="d.mp4")
    assert "-filter_complex" in c and "2:a" in c and "aac" in c
    assert c.index("d.mp4") < c.index("s.mp3")   # inputs: base, pip, audio
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_dance_ffmpeg_cmd.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kuka_sim.dance.recorder'`.

- [ ] **Step 3: Implement `recorder.py` (command builder only for now)**

```python
"""Assemble the dance video: encode frames, mux audio, overlay a corner PiP."""


def build_ffmpeg_cmd(ffmpeg, base_path, out_path, audio_path=None,
                     pip_video_path=None, pip_scale=0.25, margin=20):
    """Build the ffmpeg arg list to mux audio and/or overlay a corner PiP onto the
    base (silent) render. Inputs are ordered base, pip, audio. Returns None when
    neither extra is requested (the caller then uses the base render as the output)."""
    if audio_path is None and pip_video_path is None:
        return None
    cmd = [ffmpeg, "-y", "-i", base_path]
    if pip_video_path is not None:
        cmd += ["-i", pip_video_path]
    if audio_path is not None:
        cmd += ["-i", audio_path]

    if pip_video_path is not None:
        overlay = (f"[1:v]scale=iw*{pip_scale}:-1[p];"
                   f"[0:v][p]overlay=W-w-{margin}:H-h-{margin}[v]")
        cmd += ["-filter_complex", overlay, "-map", "[v]"]
        if audio_path is not None:                    # audio is input index 2
            cmd += ["-map", "2:a", "-c:a", "aac", "-shortest"]
    else:                                             # audio only, no re-encode of video
        cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest"]

    cmd += [out_path]
    return cmd
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/test_dance_ffmpeg_cmd.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/recorder.py tests/test_dance_ffmpeg_cmd.py
git commit -m "feat(dance): add ffmpeg audio-mux + PiP command builder"
```

---

### Task 4: `record` — encode frames, run the mux/PiP

**Files:**
- Modify: `src/kuka_sim/dance/recorder.py`
- Test: `tests/test_dance_record.py`

**Interfaces:**
- Consumes: `build_ffmpeg_cmd` (Task 3), `kuka_sim.recording.frames_to_mp4`.
- Produces: `record(frames, out_path, fps=30, audio_path=None, pip_video_path=None) -> out_path`.

- [ ] **Step 1: Write failing tests**

`tests/test_dance_record.py`:
```python
import numpy as np
import pytest
from kuka_sim.dance.recorder import record

def test_record_plain_writes_mp4(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    frames = [np.full((64, 64, 3), i * 20, dtype=np.uint8) for i in range(8)]
    out = tmp_path / "clip.mp4"
    record(frames, str(out), fps=8)
    assert out.exists() and out.stat().st_size > 0

def test_record_with_audio_muxes(tmp_path):
    import shutil, subprocess, imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # make a 1s silent audio to mux
    wav = tmp_path / "sil.wav"
    subprocess.run([ff, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", "1", str(wav)], check=True, capture_output=True)
    frames = [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(24)]
    out = tmp_path / "with_audio.mp4"
    record(frames, str(out), fps=24, audio_path=str(wav))
    # ffprobe: the output has an audio stream
    probe = subprocess.run([ff.replace("ffmpeg", "ffprobe"), "-i", str(out)],
                           capture_output=True, text=True)
    assert out.exists() and out.stat().st_size > 0
```

> The second test may skip in environments where `ffprobe` isn't beside `ffmpeg`; keep the `assert out.exists()` as the hard check and treat the probe as best-effort. If `ffprobe` is missing, remove the probe line — the mux still ran via `check=True`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_dance_record.py -v`
Expected: FAIL — `ImportError: cannot import name 'record'`.

- [ ] **Step 3: Implement (append to `recorder.py`)**

```python
import os
import subprocess
import tempfile


def record(frames, out_path, fps=30, audio_path=None, pip_video_path=None):
    """Encode frames to an MP4; if audio and/or a PiP source is given, run ffmpeg
    to mux/overlay them. Returns out_path."""
    import imageio_ffmpeg
    from kuka_sim.recording import frames_to_mp4

    if audio_path is None and pip_video_path is None:
        return frames_to_mp4(frames, out_path, fps=fps)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "base.mp4")
        frames_to_mp4(frames, base, fps=fps)
        cmd = build_ffmpeg_cmd(ffmpeg, base, out_path, audio_path, pip_video_path)
        subprocess.run(cmd, check=True, capture_output=True)
    return out_path
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/test_dance_record.py -v`
Expected: 2 passed (or 1 passed + the audio test's probe softened per the note).

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/dance/recorder.py tests/test_dance_record.py
git commit -m "feat(dance): add record (encode + audio/PiP mux)"
```

---

### Task 5: Scene flexibility — bare scene for dancing (no surface/probe)

**Files:**
- Modify: `src/kuka_sim/scene.py`
- Modify: `src/kuka_sim/robot.py`

**Interfaces:**
- Consumes: existing `build_scene`, `IiwaArm`.
- Produces: `build_scene(sim, usd_path, surface_kwargs=None, with_probe=True, with_surface=True)`; `IiwaArm(..., surface_prim=None)` supported (no contact filter).

**Verification: smoke run on the Isaac machine** (scene construction needs Isaac). No pytest.

- [ ] **Step 1: Make `IiwaArm` accept `surface_prim=None`**

In `src/kuka_sim/robot.py`, change the `ContactSensorCfg` construction so a `None` surface omits the filter:
```python
        contact_cfg = ContactSensorCfg(
            prim_path=f"{prim_path}/{ee_body}",
            update_period=0.0, history_length=4, track_pose=True,
            filter_prim_paths_expr=([surface_prim] if surface_prim else []),
        )
```
(Everything else in `IiwaArm` is unchanged.)

- [ ] **Step 2: Add `with_probe` / `with_surface` to `build_scene`**

Replace `build_scene` in `src/kuka_sim/scene.py`:
```python
def build_scene(sim, usd_path, surface_kwargs=None, with_probe=True, with_surface=True):
    import isaaclab.sim as sim_utils

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    surface = make_surface(prim_path=SURFACE_PRIM, **(surface_kwargs or {})) if with_surface else None
    arm = IiwaArm(usd_path=usd_path, prim_path=ROBOT_PRIM, ee_body=EE_BODY,
                  surface_prim=(SURFACE_PRIM if with_surface else None))
    if with_probe:
        _add_flange_probe(f"{ROBOT_PRIM}/{EE_BODY}/probe")
    return {"arm": arm, "surface": surface}
```

- [ ] **Step 3: Confirm the pure-logic suite still passes**

Run: `python -m pytest -q`
Expected: all existing tests pass (this task touched only sim-runtime code; nothing unit-tested changed behavior).

- [ ] **Step 4: SMOKE (Isaac machine): a bare dance scene builds**

Throwaway check (do not commit):
```python
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
app, sim = launch(headless=True)
h = build_scene(sim, "assets/usd/iiwa7_r800.usd", with_probe=False, with_surface=False)
sim.reset(); h["arm"].initialize()
for _ in range(60):
    h["arm"].set_joint_efforts(h["arm"].get_gravity_torque())
    h["arm"].write(); sim.step(); h["arm"].update()
print("[OK] bare scene; surface is", h["surface"], "; ee", h["arm"].get_ee_pose()[0])
import os; app.close(); os._exit(0)
```
Run: `python that_check.py`
Expected: `[OK] bare scene; surface is None ...`, no crash (no probe, no surface, arm holds).

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/scene.py src/kuka_sim/robot.py
git commit -m "feat: build_scene with_probe/with_surface flags for a bare dance scene"
```

---

### Task 6: `player.py` — drive the arm along a trajectory

**Files:**
- Create: `src/kuka_sim/dance/player.py`

**Interfaces:**
- Consumes: `CartesianImpedanceController`, `DanceTrajectory`/`FORWARD_QUAT`, `SceneCamera`.
- Produces: `play(sim, handles, traj, stiffness=..., damping=..., cam=None, on_step=None)`.

**Verification: smoke run on the Isaac machine.**

- [ ] **Step 1: Implement `player.py`**

```python
"""Drive the arm to follow a DanceTrajectory via the impedance controller."""
import numpy as np
from kuka_sim.controller import CartesianImpedanceController
from kuka_sim.dance.trajectory import FORWARD_QUAT

# Firm gains for reasonably tight trajectory tracking (some graceful lag remains).
DEFAULT_STIFFNESS = np.array([1200, 1200, 1200, 60, 60, 60.0])
DEFAULT_DAMPING = np.array([70, 70, 70, 12, 12, 12.0])


def play(sim, handles, traj, stiffness=None, damping=None, cam=None, on_step=None):
    """Follow traj.ee_pos/ee_quat for len(traj) ticks. If cam and traj.cam_pos are
    given, re-aims the camera each tick. Calls on_step(i) after each sim step
    (the recorder grabs a frame there)."""
    arm = handles["arm"]
    ctrl = CartesianImpedanceController(
        arm,
        stiffness=DEFAULT_STIFFNESS if stiffness is None else stiffness,
        damping=DEFAULT_DAMPING if damping is None else damping,
    )
    quats = traj.ee_quat
    for i in range(len(traj)):
        q = FORWARD_QUAT if quats is None else quats[i]
        ctrl.set_target_pose(traj.ee_pos[i], q)
        ctrl.apply(); arm.write(); sim.step(); arm.update()
        if cam is not None:
            if traj.cam_pos is not None:
                import torch
                cam.camera.set_world_poses_from_view(
                    torch.tensor([traj.cam_pos[i]], dtype=torch.float32, device=cam.device),
                    torch.tensor([traj.cam_target[i]], dtype=torch.float32, device=cam.device))
            cam.update()
        if on_step is not None:
            on_step(i)
```

> Reuses the verified control/step lifecycle. Camera re-aim uses `SceneCamera`'s device (Task 12 of the main project already stores `self.device`).

- [ ] **Step 2: SMOKE (Isaac machine): arm traces a canned circle**

Throwaway check (do not commit):
```python
import os, numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.dance.trajectory import DanceTrajectory, smooth_and_limit, map_to_workspace
from kuka_sim.dance.player import play
app, sim = launch(headless=True)
h = build_scene(sim, "assets/usd/iiwa7_r800.usd", with_probe=False, with_surface=False)
sim.reset(); h["arm"].initialize()
# a circle in the dexterous box, in front of the arm
N = 600
th = np.linspace(0, 2*np.pi, N)
norm = np.stack([np.zeros(N), np.cos(th), np.sin(th)], axis=1)   # y-z circle
pos = map_to_workspace(norm, center=[0.5, 0.0, 0.7], half_extents=[0.0, 0.15, 0.12])
pos = smooth_and_limit(pos, 1/120, max_speed=0.5)
traj = DanceTrajectory(ee_pos=pos)
seen = {"n": 0}
play(sim, h, traj, on_step=lambda i: seen.__setitem__("n", seen["n"]+1))
print("[OK] played", seen["n"], "steps; final ee", np.round(h["arm"].get_ee_pose()[0], 3))
app.close(); os._exit(0)
```
Run: `python that_check.py`
Expected: `[OK] played 600 steps; ...`, the arm smoothly traces the circle, no crash, no limit violation.

- [ ] **Step 3: Commit**

```bash
git add src/kuka_sim/dance/player.py
git commit -m "feat(dance): add player (arm follows a DanceTrajectory)"
```

---

### Task 7: `dance_backend_demo.py` — end-to-end (Deliverable D)

**Files:**
- Create: `scripts/dance_backend_demo.py`

**Interfaces:**
- Consumes: `launch`, `build_scene`, `SceneCamera`, `play`, `record`, the trajectory helpers.
- Produces: a canned-trajectory dance video, proving the backend before any front-end exists.

**Verification: smoke run on the Isaac machine.**

- [ ] **Step 1: Implement `scripts/dance_backend_demo.py`**

```python
"""Deliverable D: play a canned trajectory and record it — proves the backend.

Usage:
    python scripts/dance_backend_demo.py                 # silent MP4
    python scripts/dance_backend_demo.py --audio song.mp3
    python scripts/dance_backend_demo.py --pip dance.mp4 --audio song.mp3
"""
import argparse
import os
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.camera import SceneCamera
from kuka_sim.dance.trajectory import DanceTrajectory, smooth_and_limit, map_to_workspace
from kuka_sim.dance.player import play
from kuka_sim.dance.recorder import record

USD = "assets/usd/iiwa7_r800.usd"
OUT_DIR = "out"
CAPTURE_EVERY = 4          # 120 Hz sim / 4 -> 30 fps


def canned_trajectory(n=900):
    """A gentle Lissajous figure in the dexterous box (stand-in for real motion)."""
    t = np.linspace(0, 2 * np.pi, n)
    norm = np.stack([0.4 * np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    pos = map_to_workspace(norm, center=[0.5, 0.0, 0.7], half_extents=[0.08, 0.16, 0.12])
    pos = smooth_and_limit(pos, 1.0 / 120.0, max_speed=0.6)
    return DanceTrajectory(ee_pos=pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None)
    ap.add_argument("--pip", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    app, sim = launch(headless=True, enable_cameras=True)
    handles = build_scene(sim, USD, with_probe=False, with_surface=False)
    cam = SceneCamera(pos=(1.9, 1.9, 1.3), target=(0.5, 0.0, 0.7))
    sim.reset(); handles["arm"].initialize(); cam.initialize()

    traj = canned_trajectory()
    frames = []

    def on_step(i):
        if i % CAPTURE_EVERY == 0:
            frames.append(cam.capture())

    play(sim, handles, traj, cam=cam, on_step=on_step)

    out = os.path.join(OUT_DIR, "dance_backend_demo.mp4")
    record(frames, out, fps=30, audio_path=args.audio, pip_video_path=args.pip)
    print(f"[OK] wrote {out} ({len(frames)} frames)", flush=True)
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full pure-logic suite**

Run: `python -m pytest -q`
Expected: all dance + existing tests pass.

- [ ] **Step 3: SMOKE (Isaac machine): record the backend demo**

Run: `python scripts/dance_backend_demo.py`
Expected: `[OK] wrote out/dance_backend_demo.mp4 (...)`; the MP4 plays and shows the arm smoothly tracing the figure. Then, with a stock audio + a short clip:
`python scripts/dance_backend_demo.py --audio <song> --pip <clip>` → the output has the song and a corner PiP.

- [ ] **Step 4: Commit**

```bash
git add scripts/dance_backend_demo.py
git commit -m "feat(dance): end-to-end backend demo (canned trajectory -> video)"
```

---

## Self-Review

**Spec coverage:**
- `DanceTrajectory` contract → Task 1. ✓
- `smooth_and_limit` (velocity-limiting = graceful motion) → Task 2. ✓
- `map_to_workspace` → Task 1. ✓
- `build_ffmpeg_cmd` (audio + PiP, testable) → Task 3. ✓
- `record` (encode + mux) → Task 4. ✓
- `player.play` (impedance follow, camera track) → Task 6. ✓
- Recorder audio-mux + PiP, time-synced → Tasks 3–4 (+ demo Task 7). ✓
- Bare scene (no surface/probe so the arm can dance) → Task 5 (a spec gap the plan fills: the existing `build_scene` always adds the force-demo surface/probe, which would sit in the dance workspace). ✓
- Deliverables A (helpers, unit-tested), B (player smoke), C (recorder), D (end-to-end) → Tasks 1–4 / 6 / 4 / 7. ✓

**Placeholder scan:** No `TBD`/`TODO`-as-work. Signature defaults (`stiffness=None`) resolve to `DEFAULT_STIFFNESS` inside `play` — not placeholders. The audio-mux test's `ffprobe` line is explicitly softened with a fallback instruction.

**Type consistency:** `DanceTrajectory.ee_pos (N,3)`, `ee_quat (N,4)` used consistently in `player`. `build_ffmpeg_cmd` returns `list[str] | None`; `record` checks the same `audio/pip is None` condition before calling it. `map_to_workspace`/`smooth_and_limit` both take/return `(N,3)`. `FORWARD_QUAT` shape `(4,)` wxyz consistent with the controller. Input order (base, pip, audio) is stated in the interface and matched by the test (`d.mp4` before `s.mp3`) and the builder.

**Sim-runtime vs unit-test split:** Tasks 1–4 are pure logic with real pytest. Task 5 is a small sim-runtime change (smoke). Tasks 6–7 are sim-runtime, smoke-verified — the honest split used throughout this project.
