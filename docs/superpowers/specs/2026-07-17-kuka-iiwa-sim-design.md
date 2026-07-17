# Design: `kuka` — Simulated KUKA LBR iiwa 7 R800 for Isaac Sim

**Date:** 2026-07-17
**Status:** Approved (design), pending implementation plan
**Author:** Jeff (with Claude Code)

## Problem

Bhodi, a PhD candidate, runs experiments that drive a physical **KUKA LBR iiwa
7 R800**. The lab's arm is broken, blocking his work. He needs a simulated
stand-in he can drive with his own experiment code until the hardware is fixed.

His code can talk to the robot through a **plain Python API** — it does not need
ROS2. That single fact sets the whole architecture: everything runs natively on
Windows, with no ROS2, no WSL2, and no containers.

## Goal

A turnkey git repository that stands up the iiwa 7 R800 in NVIDIA Isaac Sim on
Windows and exposes a **small, clean Python control API**. Bhodi imports that
API, writes his experiment logic against it, and drives the simulated arm the
same way he would drive the real one.

A rendered MP4 of the arm being driven is a first-class deliverable: it is both
the acceptance test ("the sim works") and a shareable artifact.

## Non-goals

- **Reinforcement / imitation learning.** This is a digital twin, not a training
  environment. Isaac *Lab* is used only as the Python robotics API layer.
- **ROS2 / FRI / real-time hardware protocols.** Not needed given the plain-Python
  decision. If Bhodi's real experiment software later turns out to need ROS2, that
  is a separate follow-on (Isaac Sim's `isaacsim.ros2.bridge` supports it).
- **Committing Isaac Sim or multi-GB assets** to the repo.

## Environment & stack (verified current, stable pairing)

| Component | Choice | Why |
|---|---|---|
| OS | Windows 11 (native) | His code is plain Python; native Windows renders fine. WSL2 Docker is ruled out — Isaac Sim's RTX/Vulkan renderer does not initialize under WSL2 (officially unsupported). |
| GPU | RTX 4060 Ti, 16 GB | Ada Lovelace + RTX fully supported; ample for a single arm. |
| Isaac Sim | **5.1** | Latest *stable*. (6.0 exists but pairs only with beta Isaac Lab 3.0.) |
| Isaac Lab | **2.3.x** | Latest stable; pins Isaac Sim 5.1. Used as the `Articulation` API layer. |
| Python | **3.11** | Required by the 5.1 / 2.3 pairing. |
| Install | pip into a py3.11 venv/conda, via `setup/install.ps1` | Isaac Sim installed separately, never in the repo. |

**Robot description source:** `lbr-stack/lbr_iiwa7_r800_description` (URDF/xacro).
Chosen so the simulated arm matches the same description the real robot's ROS2
stack (`lbr_fri_ros2_stack`) uses — keeping a path open to real-robot parity later.

## Architecture

Three isolated units, each understandable and testable on its own:

```
  setup/import_iiwa.py     URDF  ─────►  USD          (build-time, run once)
                          (assets/urdf)  (assets/usd)

  src/kuka_sim/
    scene.py       builds the Isaac scene: ground, lights, loads the iiwa USD
    robot.py       IiwaArm — wraps Isaac Lab Articulation; the API Bhodi uses
    camera.py      SceneCamera — positioned to frame the arm; captures RGB frames
    sim_app.py     boots Isaac Sim (headed or headless)

  scripts/
    run_demo.py    drives the arm through a trajectory (visual sanity check)
    record_demo.py drives the arm + captures frames → out/iiwa_demo.mp4

  experiments/
    bhodi_template.py   copy-me starting point that imports IiwaArm
```

### The control API (`IiwaArm`) — the core interface

This is the contract Bhodi codes against. It hides Isaac internals entirely.

```python
arm = IiwaArm()                  # loads the iiwa into the running sim
arm.get_joint_positions()        # -> np.ndarray, 7 joint angles (rad)
arm.get_joint_velocities()       # -> np.ndarray, 7 joint velocities
arm.set_joint_targets(q)         # position control: 7 target angles
arm.get_ee_pose()                # -> (pos, quat) forward kinematics of the flange
arm.step()                       # advance the sim one physics/render tick
arm.reset()                      # return to home configuration
```

- **Control mode:** position control by default (`set_joint_targets`). Torque /
  impedance control is a documented later extension, not in scope now.
- **What it depends on:** an initialized Isaac Sim app (`sim_app.py`) and the
  imported iiwa USD (`assets/usd`).
- **What consumes it:** `run_demo.py`, `record_demo.py`, and Bhodi's experiments.
  None of them touch Isaac Lab directly.

### The camera / recording unit (`SceneCamera`)

```python
cam = SceneCamera(pos=(1.5, 1.5, 1.2), look_at=(0, 0, 0.4))
frame = cam.capture()            # -> RGB ndarray for the current sim state
```

`record_demo.py` loops `set_joint_targets → step → cam.capture`, collects frames,
and encodes them to MP4 with `imageio`/ffmpeg. Runs headless — no GUI required.

## Data flow

1. **Build time:** `import_iiwa.py` converts the vendored URDF to a USD under
   `assets/usd/` (gitignored, regenerable).
2. **Run time:** `sim_app.py` boots Isaac → `scene.py` loads the USD →
   `IiwaArm` wraps the articulation.
3. **Control loop:** caller (`run_demo` / `record_demo` / Bhodi) sets joint
   targets, steps the sim, reads state back.
4. **Recording:** `SceneCamera.capture()` pulls an RGB frame each step;
   `record_demo.py` encodes them to `out/iiwa_demo.mp4`.

## Error handling

- **Missing Isaac Sim:** `install.ps1` verifies Python 3.11 and the `isaaclab`
  import; fails loudly with remediation text if absent.
- **Missing USD:** `scene.py` checks for `assets/usd/iiwa.usd` and directs the
  user to run `import_iiwa.py` if it's not there.
- **Wrong joint-target length:** `IiwaArm.set_joint_targets` validates a length-7
  input and raises a clear `ValueError`.
- **Encoding failure:** `record_demo.py` still writes the raw frames so a run is
  never lost to a bad ffmpeg step.

## Deliverables (each independently verifiable)

- **A — Turnkey install.** Clone → `setup/install.ps1` → environment ready.
  *Verify:* `python -c "import isaaclab"` succeeds.
- **B — Arm loads.** `import_iiwa.py` produces the USD; the iiwa appears in Isaac.
  *Verify:* headed launch shows the arm standing on the ground plane.
- **C — Control API + demo.** `IiwaArm` drives the arm; `run_demo.py` moves it to
  commanded targets. *Verify:* arm visibly reaches the target configuration.
- **D — Rendered video.** `record_demo.py` writes `out/iiwa_demo.mp4` — a smooth
  clip of the arm being driven. *Verify:* the MP4 exists and plays.
- **E — Experiment template + docs.** `bhodi_template.py` + `docs/quickstart.md`
  let Bhodi start from a working example. *Verify:* the template runs unmodified.

## Open questions (non-blocking)

- **Control mode.** Confirm with Bhodi whether his experiments need torque /
  impedance control. Default is position control; impedance is a later add.
- **Real interface.** Bhodi mostly used the SmartPAD and "talked about ROS2." If
  his experiment software actually needs ROS2, revisit with the Windows-sim +
  ROS2-in-WSL2 bridge topology. Not needed under the current plain-Python plan.
