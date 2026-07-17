# Design: `kuka` — Simulated KUKA LBR iiwa 7 R800 for Isaac Sim

**Date:** 2026-07-17
**Status:** Approved (design), pending implementation plan
**Author:** Jeff (with Claude Code)

## Problem

Bhodi, a PhD candidate, runs experiments that drive a physical **KUKA LBR iiwa
7 R800**. The lab's arm is broken, blocking his work. He needs a simulated
stand-in he can drive with his own experiment code until the hardware is fixed.

His experiments are **force-controlled and contact-rich**: the arm applies a
controlled, roughly constant force against a surface (his framing: like a
surgical robot pressing/cutting on skin). The iiwa 7 R800 is a torque-sensored,
compliant robot, so **torque and Cartesian-impedance control are the right
primitives** — this matches how the real arm is used for contact tasks.

His code can drive the robot through a **plain Python API** — it does not need
ROS2. That fact sets the base architecture: everything runs natively on Windows,
with no ROS2, no WSL2, and no containers.

## Goal

A turnkey git repository that stands up the iiwa 7 R800 in NVIDIA Isaac Sim on
Windows and exposes a **Cartesian impedance / hybrid-force control API** (over a
torque-controlled articulation, with end-effector force sensing). Bhodi commands
a desired pose or contact force, reads back the measured wrench, and runs his
experiment against the sim the way he would against the real arm.

A rendered MP4 of the arm **approaching a surface and holding a target contact
force**, paired with a force-vs-time plot, is a first-class deliverable: it is
both the acceptance test and a shareable artifact.

## Non-goals

- **Reinforcement / imitation learning.** This is a digital twin, not a training
  environment. Isaac *Lab* is used only as the Python robotics API layer.
- **ROS2 / FRI / real-time hardware protocols.** Not needed given plain Python. A
  ROS2 follow-on is possible later (`isaacsim.ros2.bridge`) but out of scope.
- **Cutting / material separation / topology change.** Not native to PhysX;
  research-grade. Explicitly deferred — the surface starts *compliant*, not cut.
- **Committing Isaac Sim or multi-GB assets** to the repo.

## Environment & stack (verified current, stable pairing)

| Component | Choice | Why |
|---|---|---|
| OS | Windows 11 (native) | His code is plain Python; native Windows renders fine. WSL2 Docker is ruled out — Isaac Sim's RTX/Vulkan renderer does not initialize under WSL2 (officially unsupported). |
| GPU | RTX 4060 Ti, 16 GB | Ada Lovelace + RTX fully supported; ample for a single arm with contact. |
| Isaac Sim | **5.1** | Latest *stable*. (6.0 pairs only with beta Isaac Lab 3.0.) |
| Isaac Lab | **2.3.x** | Latest stable; pins Isaac Sim 5.1. Used for the `Articulation` API, effort control, Jacobians, and contact/force sensing. |
| Python | **3.11** | Required by the 5.1 / 2.3 pairing. |
| Install | pip into a py3.11 venv/conda, via `setup/install.ps1` | Isaac Sim installed separately, never in the repo. |

**Robot description source:** `lbr-stack/lbr_iiwa7_r800_description` (URDF/xacro),
imported to USD. Chosen so the arm matches the description the real robot's ROS2
stack uses — keeping a path open to real-robot parity later.

**To verify during planning:** exact API for (a) effort/torque control on an Isaac
Lab `Articulation` in 2.3, (b) end-effector contact wrench readback (contact
sensor vs. articulation force sensor), and (c) tunable contact stiffness/damping
for the compliant surface. These are known-supported; the precise call sites get
confirmed against 2.3 docs when the plan is written.

## Architecture

Units are kept small and swappable — especially the surface, since Bhodi's exact
task is not 100% pinned and the surface may later become deformable.

```
  setup/import_iiwa.py      URDF ─────► USD           (build-time, run once)

  src/kuka_sim/
    scene.py        builds the scene: ground, lights, iiwa, and a Surface
    surface.py      CompliantSurface — a stiffness/damping-tunable contact body
                    (modular; a DeformableSurface can replace it later)
    robot.py        IiwaArm — torque-controlled Articulation + EE wrench sensing
    controller.py   CartesianImpedanceController — desired pose/force -> joint torques
    camera.py       SceneCamera — frames the arm; captures RGB frames
    sim_app.py      boots Isaac Sim (headed or headless)

  scripts/
    run_demo.py     free-space impedance move (no contact) — first sanity check
    force_demo.py   approach surface + hold target contact force; logs force
    record_demo.py  runs force_demo + captures frames -> out/iiwa_force_demo.mp4
                    and a force-vs-time plot out/force_trace.png

  experiments/
    bhodi_template.py   copy-me example: command a force, read the wrench, log it
```

### Low-level robot interface (`IiwaArm`)

Torque-controlled articulation plus sensing. Bhodi *can* use this directly, but
usually goes through the controller.

```python
arm = IiwaArm()                  # loads the iiwa into the running sim
arm.get_joint_positions()        # -> np.ndarray(7)  angles (rad)
arm.get_joint_velocities()       # -> np.ndarray(7)
arm.set_joint_efforts(tau)       # 7 joint torques (N·m) — effort control
arm.get_ee_pose()                # -> (pos, quat) flange forward kinematics
arm.get_ee_wrench()              # -> (force[3], torque[3]) measured EE contact wrench
arm.get_jacobian()               # -> 6x7 EE Jacobian (for the controller)
arm.step(); arm.reset()
```

### Primary control layer (`CartesianImpedanceController`) — what Bhodi uses

Implements Cartesian impedance / hybrid force control on top of joint torques
(gravity compensation + Jacobian-transpose mapping). This is the closest analog
to how the real iiwa is driven for contact tasks.

```python
ctrl = CartesianImpedanceController(arm,
                                    stiffness=(kx, ky, kz, ...),  # per-axis
                                    damping=...)
ctrl.set_target_pose(pos, quat)      # move/hold a pose compliantly (free space)
ctrl.set_target_wrench(force)        # apply/hold a desired force, e.g. -N along surface normal
tau = ctrl.compute()                 # -> joint torques for this tick
ctrl.apply()                         # compute() then arm.set_joint_efforts(tau)
```

- **What it depends on:** `IiwaArm` (state, Jacobian, effort command) and a booted
  sim. **What consumes it:** the demos and Bhodi's experiments.
- **Hybrid force/motion:** position-controlled in free axes, force-controlled along
  the contact normal — the mode a "press with constant force" task needs.

### Surface (`CompliantSurface`) — deliberately swappable

Starts as a rigid plate with **tunable contact stiffness/damping** (a
spring-backed compliant contact), giving realistic force-regulation dynamics
without soft-body cost. Exposes the same interface a future `DeformableSurface`
(FEM soft body) would, so upgrading fidelity is a contained change.

### Camera / recording (`SceneCamera`)

`cam.capture()` returns an RGB frame for the current sim state. `record_demo.py`
loops the force demo, captures frames, and encodes MP4 via `imageio`/ffmpeg.
Runs headless — no GUI required.

## Data flow

1. **Build time:** `import_iiwa.py` converts the vendored URDF to USD under
   `assets/usd/` (gitignored, regenerable).
2. **Run time:** `sim_app.py` boots Isaac → `scene.py` loads the iiwa + surface →
   `IiwaArm` wraps the articulation → `CartesianImpedanceController` wraps the arm.
3. **Control loop each tick:** read state (pose, wrench, Jacobian) →
   `ctrl.compute()` → `arm.set_joint_efforts(tau)` → `arm.step()`.
4. **Recording:** `SceneCamera.capture()` pulls a frame per step; `record_demo.py`
   encodes `out/iiwa_force_demo.mp4` and plots the logged force trace.

## Error handling

- **Missing Isaac Sim:** `install.ps1` verifies Python 3.11 and the `isaaclab`
  import; fails loudly with remediation text.
- **Missing USD:** `scene.py` checks for `assets/usd/iiwa.usd` and points to
  `import_iiwa.py` if absent.
- **Wrong torque/target length:** `set_joint_efforts` validates length 7;
  controller validates target shapes; clear `ValueError`s.
- **Torque / instability guard:** the controller clamps commanded joint torques to
  the iiwa's rated limits so a bad gain can't launch the arm; large steps warn.
- **Encoding failure:** `record_demo.py` still writes raw frames + the force log so
  a run is never lost to a bad ffmpeg step.

## Deliverables (each independently verifiable)

- **A — Turnkey install.** Clone → `setup/install.ps1` → environment ready.
  *Verify:* `python -c "import isaaclab"` succeeds.
- **B — Arm loads.** `import_iiwa.py` produces the USD; the iiwa appears in Isaac.
  *Verify:* headed launch shows the arm on the ground plane.
- **C — Torque control + impedance (free space).** `run_demo.py` holds/moves a
  Cartesian pose compliantly via joint torques. *Verify:* arm reaches and holds
  the target pose; pushing it (scripted disturbance) shows compliance.
- **D — Force regulation on the compliant surface.** `force_demo.py` approaches the
  surface and holds a target contact force. *Verify:* measured EE force converges
  to the target (within tolerance) in the logged trace.
- **E — Rendered video + force plot.** `record_demo.py` writes
  `out/iiwa_force_demo.mp4` (arm pressing the surface) and `out/force_trace.png`
  (force vs. time). *Verify:* the MP4 plays and the plot shows force holding.
- **F — Experiment template + docs.** `bhodi_template.py` + `docs/quickstart.md`
  let Bhodi command a force, read the wrench, and log — from a working example.
  *Verify:* the template runs unmodified.

## Open questions (non-blocking)

- **Exact task.** Not 100% confirmed that Bhodi is doing constant-force-on-
  malleable-surface. The compliant surface + Cartesian force controller cover the
  likely case; the surface is swappable if the task turns out different.
- **Surface fidelity.** Starting compliant (rigid + contact stiffness). If he needs
  visible deformation, upgrade `CompliantSurface` → an FEM `DeformableSurface`
  (supported, heavier). Cutting/material removal remains out of scope.
- **Tool geometry.** Whether a specific end-effector tool (scalpel/probe) must be
  mounted on the flange, or a generic tip is fine for now. Default: generic tip.
