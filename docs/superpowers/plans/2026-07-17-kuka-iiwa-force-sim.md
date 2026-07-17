# KUKA iiwa 7 R800 Force-Control Sim — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a simulated KUKA LBR iiwa 7 R800 in Isaac Sim on Windows that Bhodi drives via a Cartesian impedance / hybrid-force Python API, with a flagship demo that presses a compliant surface at a target force and renders an MP4 + force plot.

**Architecture:** A torque-controlled Isaac Lab `Articulation` is wrapped by `IiwaArm` (implements a small `ArmInterface`). A provider-agnostic `CartesianImpedanceController` consumes that interface and maps pose/force targets to clamped joint torques via `Jᵀ·F` + gravity compensation. All controller math is unit-tested against a `FakeArm`; the Isaac-runtime glue (arm, surface, scene, camera, URDF import) is verified by smoke scripts on the Isaac machine. Demos and Bhodi's experiments consume only `IiwaArm` + the controller, never Isaac internals.

**Tech Stack:** Isaac Sim 5.1, Isaac Lab 2.3.x, Python 3.11, PyTorch (bundled with Isaac), NumPy, Matplotlib (Agg), imageio + imageio-ffmpeg, pytest.

## Global Constraints

- **Python 3.11 exactly** — required by Isaac Sim 5.1 / Isaac Lab 2.3.
- **Isaac Sim 5.1 + Isaac Lab 2.3.x** — the stable pairing. Do NOT use Isaac Sim 6.0 / Isaac Lab 3.0 (beta).
- **Native Windows 11** — no WSL2, no Docker for the sim (RTX/Vulkan does not initialize under WSL2).
- **Isaac import namespace is flat `isaaclab.*`** (the old `omni.isaac.lab.*` is retired).
- **Isaac Sim / Isaac Lab are installed separately** by `setup/install.ps1`; they are NEVER committed and NEVER listed in `pyproject.toml` runtime deps.
- **Generated artifacts are gitignored:** `assets/usd/`, `out/`, `*.mp4`, `*.usd*` (already in `.gitignore`).
- **Torque control requires `ImplicitActuatorCfg(stiffness=0.0, damping=0.0)`** so commanded efforts pass through with no hidden PD term.
- **iiwa 7 R800 rated joint torque limits (N·m):** `[176, 176, 110, 110, 110, 40, 40]` — the controller clamps to these.
- **Quaternions are `wxyz`** (Isaac Lab convention) everywhere.
- **Effort command shape** to `set_joint_effort_target` is `(num_envs, num_joints)` = `(1, 7)`; must be followed by `robot.write_data_to_sim()`.
- **RGB capture requires `AppLauncher(enable_cameras=True)`** or no frames are produced headless.

### Interface contract (used across tasks)

`ArmInterface` — the 5 methods the controller consumes (Task 3), implemented by both `FakeArm` (test double) and `IiwaArm` (real, Task 10):

```
get_joint_velocities() -> np.ndarray shape (7,)
get_ee_pose()          -> tuple(pos np.ndarray (3,), quat np.ndarray (4,) wxyz)
get_jacobian()         -> np.ndarray shape (6, 7), base frame
get_gravity_torque()   -> np.ndarray shape (7,)
set_joint_efforts(tau: np.ndarray shape (7,)) -> None
```

`IiwaArm` additionally provides (for demos, not the controller): `get_joint_positions() -> (7,)`, `get_ee_force() -> (3,)` (contact force, world frame), `write()`, `update()`, `reset()`.

`CartesianImpedanceController` (Task 4) public surface:
```
__init__(arm, stiffness (6,), damping (6,), torque_limits=(7,)|None)
set_target_pose(pos (3,), quat (4,)) -> None
set_target_wrench(wrench (6,), selection (6,) bool) -> None
compute() -> np.ndarray (7,)   # clamped joint torques
apply() -> None                # arm.set_joint_efforts(compute())
```

> **Deviation from spec:** the spec named `get_ee_wrench()` (6-DOF). Isaac's `ContactSensor` exposes 3-DOF force only (no contact torque), so the arm exposes `get_ee_force() -> (3,)`. This is sufficient for normal-force regulation. The spec is amended to match.

---

### Task 1: Project scaffold & packaging

**Files:**
- Create: `pyproject.toml`
- Create: `src/kuka_sim/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `kuka_sim` package and a working `pytest` harness for all later tasks.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "kuka_sim"
version = "0.1.0"
description = "Simulated KUKA LBR iiwa 7 R800 (Isaac Sim) with Cartesian force control"
requires-python = "==3.11.*"
dependencies = [
    "numpy>=1.24",
    "matplotlib>=3.7",
    "imageio>=2.31",
    "imageio-ffmpeg>=0.4.9",
]
# NOTE: isaacsim / isaaclab are installed separately by setup/install.ps1 — never here.

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the package + test package init files**

`src/kuka_sim/__init__.py`:
```python
"""Simulated KUKA LBR iiwa 7 R800 with Cartesian force control."""
__version__ = "0.1.0"
```

`tests/__init__.py`: (empty file)

- [ ] **Step 3: Write the smoke test**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import kuka_sim
    assert kuka_sim.__version__ == "0.1.0"
```

- [ ] **Step 4: Run the smoke test — expect PASS**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write `README.md`** (short; quickstart doc comes in Task 15)

```markdown
# kuka — Simulated KUKA LBR iiwa 7 R800

A digital twin of the KUKA LBR iiwa 7 R800 in NVIDIA Isaac Sim, driven by a
Cartesian impedance / hybrid-force Python API. Built so Bhodi can run his
force-controlled experiments in sim while the physical arm is down.

See `docs/superpowers/plans/2026-07-17-kuka-iiwa-force-sim.md` for the build plan
and `docs/quickstart.md` (added in the final step) for setup and usage.

## Layout
- `src/kuka_sim/` — the package (arm, controller, surface, scene, camera, utils)
- `scripts/` — runnable demos (`run_demo`, `force_demo`, `record_demo`)
- `experiments/` — Bhodi's starting template
- `setup/` — install + URDF-import scripts
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/kuka_sim/__init__.py tests/__init__.py tests/test_smoke.py README.md
git commit -m "chore: scaffold kuka_sim package and pytest harness"
```

---

### Task 2: Quaternion math utilities

**Files:**
- Create: `src/kuka_sim/math_utils.py`
- Test: `tests/test_math_utils.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `quat_conj(q)`, `quat_mul(a, b)`, `orientation_error(q_des, q_cur) -> (3,)` — used by the controller (Task 4).

- [ ] **Step 1: Write failing tests**

`tests/test_math_utils.py`:
```python
import numpy as np
from kuka_sim.math_utils import quat_conj, quat_mul, orientation_error

def test_quat_mul_identity():
    q = np.array([0.0, 0.0, 1.0, 0.0])          # wxyz
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_mul(ident, q), q)

def test_quat_conj():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(quat_conj(q), [1.0, -2.0, -3.0, -4.0])

def test_orientation_error_zero_when_equal():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(orientation_error(q, q), np.zeros(3))

def test_orientation_error_90deg_about_z():
    c = np.cos(np.pi / 4); s = np.sin(np.pi / 4)
    q_des = np.array([c, 0.0, 0.0, s])           # +90° about z
    q_cur = np.array([1.0, 0.0, 0.0, 0.0])
    err = orientation_error(q_des, q_cur)
    assert np.allclose(err, [0.0, 0.0, np.pi / 2], atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_math_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kuka_sim.math_utils'`.

- [ ] **Step 3: Implement `math_utils.py`**

```python
"""Quaternion helpers (wxyz convention). Pure NumPy — no Isaac dependency."""
import numpy as np


def quat_conj(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    aw, ax, ay, az = np.asarray(a, dtype=float)
    bw, bx, by, bz = np.asarray(b, dtype=float)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def orientation_error(q_des, q_cur):
    """Axis-angle (3,) rotation taking q_cur -> q_des, in the current frame."""
    q_err = quat_mul(np.asarray(q_des, float), quat_conj(q_cur))
    if q_err[0] < 0.0:                    # shortest path
        q_err = -q_err
    vec = q_err[1:]
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(norm, q_err[0])
    return vec / norm * angle
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/test_math_utils.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/math_utils.py tests/test_math_utils.py
git commit -m "feat: add quaternion math utils for orientation error"
```

---

### Task 3: Arm interface + FakeArm test double

**Files:**
- Create: `src/kuka_sim/interfaces.py`
- Create: `tests/fakes.py`
- Test: `tests/test_fakes.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ArmInterface` (Protocol) and `FakeArm` — the deterministic test double the controller tests (Task 4) drive. `FakeArm(jacobian, gravity, ee_pose, joint_vel)` records the last `set_joint_efforts` call in `.applied`.

- [ ] **Step 1: Write failing test**

`tests/test_fakes.py`:
```python
import numpy as np
from tests.fakes import FakeArm

def test_fakearm_defaults_and_records_efforts():
    arm = FakeArm()
    assert arm.get_jacobian().shape == (6, 7)
    assert arm.get_gravity_torque().shape == (7,)
    pos, quat = arm.get_ee_pose()
    assert pos.shape == (3,) and quat.shape == (4,)
    assert arm.get_joint_velocities().shape == (7,)
    arm.set_joint_efforts(np.arange(7.0))
    assert np.allclose(arm.applied, np.arange(7.0))

def test_fakearm_rejects_bad_effort_length():
    arm = FakeArm()
    try:
        arm.set_joint_efforts(np.zeros(6))
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_fakes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fakes'`.

- [ ] **Step 3: Implement `interfaces.py`**

```python
"""The minimal arm contract the controller depends on."""
from typing import Protocol, Tuple
import numpy as np


class ArmInterface(Protocol):
    def get_joint_velocities(self) -> np.ndarray: ...           # (7,)
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]: ...  # (3,), (4,) wxyz
    def get_jacobian(self) -> np.ndarray: ...                   # (6, 7) base frame
    def get_gravity_torque(self) -> np.ndarray: ...             # (7,)
    def set_joint_efforts(self, tau: np.ndarray) -> None: ...   # (7,)
```

- [ ] **Step 4: Implement `tests/fakes.py`**

```python
"""Deterministic in-memory arm for controller unit tests."""
import numpy as np


class FakeArm:
    def __init__(self, jacobian=None, gravity=None, ee_pose=None, joint_vel=None):
        # default Jacobian: 6x7 with 1s on the leading diagonal (task axis i <- joint i)
        self._J = np.eye(6, 7) if jacobian is None else np.asarray(jacobian, float)
        self._g = np.zeros(7) if gravity is None else np.asarray(gravity, float)
        if ee_pose is None:
            self._pos, self._quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self._pos, self._quat = (np.asarray(ee_pose[0], float),
                                     np.asarray(ee_pose[1], float))
        self._qd = np.zeros(7) if joint_vel is None else np.asarray(joint_vel, float)
        self.applied = None

    def get_joint_velocities(self):
        return self._qd

    def get_ee_pose(self):
        return self._pos, self._quat

    def get_jacobian(self):
        return self._J

    def get_gravity_torque(self):
        return self._g

    def set_joint_efforts(self, tau):
        tau = np.asarray(tau, float)
        if tau.shape != (7,):
            raise ValueError(f"tau must be shape (7,), got {tau.shape}")
        self.applied = tau
```

- [ ] **Step 5: Run to verify PASS**

Run: `python -m pytest tests/test_fakes.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/kuka_sim/interfaces.py tests/fakes.py tests/test_fakes.py
git commit -m "feat: add ArmInterface protocol and FakeArm test double"
```

---

### Task 4: Cartesian impedance / hybrid-force controller

**Files:**
- Create: `src/kuka_sim/controller.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: `ArmInterface` (Task 3), `orientation_error` (Task 2).
- Produces: `CartesianImpedanceController` and `clamp_torque(tau, limits)`, plus module constant `IIWA_TORQUE_LIMITS = np.array([176,176,110,110,110,40,40])`.

- [ ] **Step 1: Write failing tests**

`tests/test_controller.py`:
```python
import numpy as np
import pytest
from kuka_sim.controller import CartesianImpedanceController, IIWA_TORQUE_LIMITS
from tests.fakes import FakeArm

K = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
D = np.zeros(6)

def _ctrl(arm, k=K, d=D):
    c = CartesianImpedanceController(arm, stiffness=k, damping=d)
    return c

def test_zero_error_zero_torque():
    arm = FakeArm()                                  # ee at origin, identity quat
    c = _ctrl(arm)
    c.set_target_pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
    assert np.allclose(c.compute(), np.zeros(7))

def test_position_error_maps_through_jacobian_transpose():
    arm = FakeArm()                                  # J = eye(6,7), gravity 0
    c = _ctrl(arm)
    c.set_target_pose(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0, 0, 0]))
    tau = c.compute()
    # x_err = [1,0,0,0,0,0]; F = K*x_err = [10,0,...]; tau = J^T F
    assert np.allclose(tau, [10, 0, 0, 0, 0, 0, 0])

def test_gravity_compensation_added():
    g = np.arange(1.0, 8.0)
    arm = FakeArm(gravity=g)
    c = _ctrl(arm)
    c.set_target_pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
    assert np.allclose(c.compute(), g)

def test_damping_opposes_velocity():
    qd = np.array([0.5, 0, 0, 0, 0, 0, 0])
    arm = FakeArm(joint_vel=qd)
    c = _ctrl(arm, d=np.array([2.0, 0, 0, 0, 0, 0, 0]))
    c.set_target_pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
    # x_dot = J qd = [0.5,0,...]; F = -D*x_dot = [-1,0,...]; tau = J^T F
    assert np.allclose(c.compute(), [-1.0, 0, 0, 0, 0, 0, 0])

def test_force_axis_override():
    arm = FakeArm()
    c = _ctrl(arm)
    c.set_target_pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
    sel = np.array([False, False, True, False, False, False])
    c.set_target_wrench(np.array([0, 0, -5.0, 0, 0, 0]), sel)
    tau = c.compute()
    # z-axis forced to -5 regardless of impedance; tau[2] = -5
    assert np.isclose(tau[2], -5.0)

def test_torque_clamped_to_limits():
    arm = FakeArm()
    c = CartesianImpedanceController(arm, stiffness=np.full(6, 1e6), damping=np.zeros(6))
    c.set_target_pose(np.array([100.0, 0, 0]), np.array([1.0, 0, 0, 0]))
    tau = c.compute()
    assert np.all(np.abs(tau) <= IIWA_TORQUE_LIMITS + 1e-9)
    assert np.isclose(tau[0], IIWA_TORQUE_LIMITS[0])

def test_compute_requires_target_pose():
    with pytest.raises(RuntimeError):
        _ctrl(FakeArm()).compute()

def test_bad_gain_shape_rejected():
    with pytest.raises(ValueError):
        CartesianImpedanceController(FakeArm(), stiffness=np.zeros(5), damping=np.zeros(6))

def test_apply_sends_efforts_to_arm():
    arm = FakeArm(gravity=np.arange(1.0, 8.0))
    c = _ctrl(arm)
    c.set_target_pose(np.zeros(3), np.array([1.0, 0, 0, 0]))
    c.apply()
    assert np.allclose(arm.applied, np.arange(1.0, 8.0))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_controller.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kuka_sim.controller'`.

- [ ] **Step 3: Implement `controller.py`**

```python
"""Cartesian impedance / hybrid-force controller over a torque-controlled arm."""
import numpy as np
from kuka_sim.math_utils import orientation_error

# iiwa 7 R800 rated joint torque limits (N·m), joints 1..7
IIWA_TORQUE_LIMITS = np.array([176.0, 176.0, 110.0, 110.0, 110.0, 40.0, 40.0])


def clamp_torque(tau, limits):
    return np.clip(np.asarray(tau, float), -np.asarray(limits, float), np.asarray(limits, float))


class CartesianImpedanceController:
    """Maps a Cartesian pose/force target to clamped joint torques.

    tau = J^T ( K * x_err - D * (J qdot) )  + gravity,  with force-selected
    task axes replaced by a feed-forward wrench (hybrid force/motion control).
    """

    def __init__(self, arm, stiffness, damping, torque_limits=None):
        self.arm = arm
        self.K = np.asarray(stiffness, float)
        self.D = np.asarray(damping, float)
        if self.K.shape != (6,) or self.D.shape != (6,):
            raise ValueError("stiffness and damping must both be shape (6,)")
        self.limits = (IIWA_TORQUE_LIMITS if torque_limits is None
                       else np.asarray(torque_limits, float))
        self._target_pos = None
        self._target_quat = None
        self._ff_wrench = np.zeros(6)
        self._force_sel = np.zeros(6, dtype=bool)

    def set_target_pose(self, pos, quat):
        pos = np.asarray(pos, float)
        quat = np.asarray(quat, float)
        if pos.shape != (3,) or quat.shape != (4,):
            raise ValueError("pos must be (3,) and quat (4,)")
        self._target_pos, self._target_quat = pos, quat

    def set_target_wrench(self, wrench, selection):
        w = np.asarray(wrench, float)
        s = np.asarray(selection, bool)
        if w.shape != (6,) or s.shape != (6,):
            raise ValueError("wrench and selection must both be shape (6,)")
        self._ff_wrench, self._force_sel = w, s

    def compute(self):
        if self._target_pos is None:
            raise RuntimeError("call set_target_pose(...) before compute()")
        pos, quat = self.arm.get_ee_pose()
        pos_err = self._target_pos - np.asarray(pos, float)
        ori_err = orientation_error(self._target_quat, quat)
        x_err = np.concatenate([pos_err, ori_err])          # (6,)
        J = np.asarray(self.arm.get_jacobian(), float)       # (6, 7)
        qd = np.asarray(self.arm.get_joint_velocities(), float)
        x_dot = J @ qd                                       # (6,)
        F = self.K * x_err - self.D * x_dot                  # (6,) impedance wrench
        F = np.where(self._force_sel, self._ff_wrench, F)    # hybrid override
        tau = J.T @ F + np.asarray(self.arm.get_gravity_torque(), float)
        return clamp_torque(tau, self.limits)

    def apply(self):
        self.arm.set_joint_efforts(self.compute())
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/test_controller.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/controller.py tests/test_controller.py
git commit -m "feat: add Cartesian impedance/hybrid-force controller"
```

---

### Task 5: Force logging + plot

**Files:**
- Create: `src/kuka_sim/logging_utils.py`
- Test: `tests/test_logging_utils.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ForceLog` with `append(t, force3)`, `as_arrays() -> (t (N,), f (N,3))`, `save_plot(path, target=None)`.

- [ ] **Step 1: Write failing tests**

`tests/test_logging_utils.py`:
```python
import numpy as np
from kuka_sim.logging_utils import ForceLog

def test_log_accumulates_and_shapes():
    log = ForceLog()
    log.append(0.0, [0, 0, -1.0])
    log.append(0.1, [0, 0, -4.0])
    t, f = log.as_arrays()
    assert t.shape == (2,) and f.shape == (2, 3)
    assert np.isclose(t[1], 0.1)

def test_save_plot_writes_file(tmp_path):
    log = ForceLog()
    for i in range(5):
        log.append(i * 0.1, [0, 0, -float(i)])
    out = tmp_path / "force.png"
    log.save_plot(str(out), target=5.0)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_logging_utils.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `logging_utils.py`**

```python
"""Contact-force logging and a force-vs-time plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless-safe; no display needed
import matplotlib.pyplot as plt


class ForceLog:
    def __init__(self):
        self._t = []
        self._f = []

    def append(self, t, force):
        self._t.append(float(t))
        self._f.append(np.asarray(force, float).reshape(3).copy())

    def as_arrays(self):
        return np.asarray(self._t), np.asarray(self._f)

    def save_plot(self, path, target=None):
        t, f = self.as_arrays()
        mag = np.linalg.norm(f, axis=1) if f.ndim == 2 else f
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, mag, label="|F| measured")
        if target is not None:
            ax.axhline(target, ls="--", color="r", label="target")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("contact force (N)")
        ax.set_title("End-effector contact force")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/test_logging_utils.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/logging_utils.py tests/test_logging_utils.py
git commit -m "feat: add force logging and force-vs-time plot"
```

---

### Task 6: Frame → MP4 encoding

**Files:**
- Create: `src/kuka_sim/recording.py`
- Test: `tests/test_recording.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `frames_to_mp4(frames, path, fps=30) -> path`. Accepts uint8 or float [0,1] frames, RGB or RGBA (alpha dropped).

- [ ] **Step 1: Write failing test**

`tests/test_recording.py`:
```python
import numpy as np
import pytest
from kuka_sim.recording import frames_to_mp4

def test_frames_to_mp4_writes_file(tmp_path):
    pytest.importorskip("imageio_ffmpeg")   # skip if the ffmpeg backend is absent
    frames = [np.full((64, 64, 3), i * 20, dtype=np.uint8) for i in range(10)]
    out = tmp_path / "clip.mp4"
    frames_to_mp4(frames, str(out), fps=10)
    assert out.exists() and out.stat().st_size > 0

def test_frames_to_mp4_handles_rgba_and_float(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    frames = [np.ones((32, 32, 4), dtype=np.float32) * 0.5 for _ in range(5)]
    out = tmp_path / "clip2.mp4"
    frames_to_mp4(frames, str(out), fps=5)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_recording.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `recording.py`**

```python
"""Encode a sequence of RGB frames to an MP4."""
import numpy as np


def frames_to_mp4(frames, path, fps=30):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (255.0 * np.clip(arr, 0.0, 1.0)).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[-1] == 4:   # drop alpha
                arr = arr[..., :3]
            writer.append_data(arr)
    return path
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/test_recording.py -v`
Expected: 2 passed (or skipped if `imageio_ffmpeg` missing — install it via `pip install imageio-ffmpeg`).

- [ ] **Step 5: Commit**

```bash
git add src/kuka_sim/recording.py tests/test_recording.py
git commit -m "feat: add frame-to-mp4 encoding"
```

---

### Task 7: Environment setup script + version check

**Files:**
- Create: `setup/check_env.py`
- Create: `setup/install.ps1`
- Test: `tests/test_check_env.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `python_version_ok(version_info=None, required=(3, 11)) -> bool` and a manual install script.

- [ ] **Step 1: Write failing test**

`tests/test_check_env.py`:
```python
from setup.check_env import python_version_ok

def test_version_ok_for_311():
    assert python_version_ok((3, 11, 5)) is True

def test_version_not_ok_for_312():
    assert python_version_ok((3, 12, 0)) is False

def test_version_not_ok_for_310():
    assert python_version_ok((3, 10, 9)) is False
```

Add `setup/__init__.py` (empty) so the test can import it, and ensure `pytest.ini_options.pythonpath` already includes the repo root implicitly (tests run from root). If import fails, add `"."` to `pythonpath` in `pyproject.toml`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_check_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'setup.check_env'`.

- [ ] **Step 3: Implement `setup/check_env.py`**

```python
"""Environment preflight for the KUKA iiwa sim."""
import sys


def python_version_ok(version_info=None, required=(3, 11)):
    v = version_info if version_info is not None else sys.version_info
    return (v[0], v[1]) == required


def main():
    if not python_version_ok():
        print(f"[FAIL] Python {sys.version_info.major}.{sys.version_info.minor} "
              f"detected; Isaac Sim 5.1 / Isaac Lab 2.3 require exactly 3.11.")
        return 1
    try:
        import isaaclab  # noqa: F401
        print("[OK] Python 3.11 and isaaclab import succeeded.")
        return 0
    except ImportError:
        print("[FAIL] isaaclab not importable. Run setup/install.ps1 first.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Create empty `setup/__init__.py`.

- [ ] **Step 4: Write `setup/install.ps1`** (manual-run; not unit tested)

```powershell
# setup/install.ps1 — one-time environment setup for the KUKA iiwa sim (Windows, native).
# Requires: Python 3.11 on PATH, an NGC account for Isaac assets, a recent NVIDIA driver.
$ErrorActionPreference = "Stop"

Write-Host "== Creating Python 3.11 venv (env_isaaclab) =="
py -3.11 -m venv env_isaaclab
.\env_isaaclab\Scripts\Activate.ps1

python -c "import sys; assert sys.version_info[:2]==(3,11), 'need Python 3.11'"

Write-Host "== Upgrading pip and installing Isaac Sim 5.1 + Isaac Lab 2.3 =="
python -m pip install --upgrade pip
# Isaac Sim 5.1 (pip) — pulls the RTX runtime; ~large download.
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# Isaac Lab 2.3 (stable) as the robotics API layer.
pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

Write-Host "== Installing project (editable) + dev deps =="
pip install -e ".[dev]"

Write-Host "== Preflight =="
python setup/check_env.py

Write-Host "== Done. Next: python setup/import_iiwa.py to build the arm USD. =="
```

> Confirm the exact `isaacsim` pip extras/tag against the Isaac Sim 5.1 pip docs at execution time; the version pin `5.1.0` and `isaaclab==2.3.2.post1` were verified during planning but pip extras occasionally shift across point releases.

- [ ] **Step 5: Run to verify PASS**

Run: `python -m pytest tests/test_check_env.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add setup/__init__.py setup/check_env.py setup/install.ps1 tests/test_check_env.py
git commit -m "feat: add environment preflight and install script"
```

---

### Task 8: URDF → USD import script

**Files:**
- Create: `setup/import_iiwa.py`
- Create: `docs/assets.md` (where the URDF comes from)

**Interfaces:**
- Consumes: Isaac Lab `UrdfConverter` (runtime).
- Produces: `assets/usd/iiwa7_r800.usd` (gitignored) from a vendored URDF. Exposes `resolve_paths(repo_root) -> (urdf_path, usd_dir, usd_name)` (path logic, no Isaac import) for a lightweight test.

**Verification is a smoke run on the Isaac machine** (needs Isaac Sim). There is no pytest for the conversion itself.

- [ ] **Step 1: Document + vendor the URDF source**

`docs/assets.md`:
```markdown
# Robot description

The iiwa 7 R800 URDF/meshes come from `lbr-stack/lbr_iiwa7_r800_description`
(chosen for parity with the real robot's ROS2 stack).

Fetch into `assets/urdf/` (run once, on any machine):

    git clone --depth 1 https://github.com/lbr-stack/lbr_iiwa7_r800_description \
        assets/urdf/lbr_iiwa7_r800_description

The description's top-level URDF/xacro is then imported to USD by
`setup/import_iiwa.py`. If the package ships xacro only, expand it first:

    xacro assets/urdf/lbr_iiwa7_r800_description/urdf/iiwa7_r800.urdf.xacro \
        > assets/urdf/iiwa7_r800.urdf

Confirm the exact file name inside the cloned repo; adjust `URDF_REL` in
`setup/import_iiwa.py` to match.
```

- [ ] **Step 2: Implement `setup/import_iiwa.py`**

```python
"""Convert the iiwa 7 R800 URDF to USD via Isaac Lab's UrdfConverter.

Run on the Isaac machine after setup/install.ps1:
    python setup/import_iiwa.py
"""
import os

# Path relative to repo root — adjust to the actual file inside the cloned description.
URDF_REL = "assets/urdf/iiwa7_r800.urdf"
USD_DIR_REL = "assets/usd"
USD_NAME = "iiwa7_r800.usd"


def resolve_paths(repo_root):
    """Pure path logic (no Isaac import) so it is unit-testable."""
    urdf = os.path.join(repo_root, URDF_REL)
    usd_dir = os.path.join(repo_root, USD_DIR_REL)
    return urdf, usd_dir, USD_NAME


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf, usd_dir, usd_name = resolve_paths(repo_root)
    if not os.path.exists(urdf):
        raise FileNotFoundError(
            f"URDF not found at {urdf}. See docs/assets.md to fetch the description.")
    os.makedirs(usd_dir, exist_ok=True)

    # Isaac imports must follow the app launch.
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    app = launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    cfg = UrdfConverterCfg(
        asset_path=urdf,
        usd_dir=usd_dir,
        usd_file_name=usd_name,
        fix_base=True,                 # iiwa is a fixed-base arm
        merge_fixed_joints=True,
        collider_type="convex_hull",
        self_collision=False,
    )
    converter = UrdfConverter(cfg)
    print(f"[OK] wrote USD: {converter.usd_path}")
    app.close()


if __name__ == "__main__":
    main()
```

> The `UrdfConverterCfg` base fields (`asset_path`, `usd_dir`, `usd_file_name`) and `converter.usd_path` were flagged during planning as needing confirmation against the installed 2.3 base class. Verify at execution; if names differ, adjust here only.

- [ ] **Step 3: Add a path-logic test**

`tests/test_import_paths.py`:
```python
import os
from setup.import_iiwa import resolve_paths

def test_resolve_paths():
    urdf, usd_dir, name = resolve_paths("/repo")
    assert urdf == os.path.join("/repo", "assets/urdf/iiwa7_r800.urdf")
    assert usd_dir == os.path.join("/repo", "assets/usd")
    assert name == "iiwa7_r800.usd"
```

- [ ] **Step 4: Run the path test — expect PASS**

Run: `python -m pytest tests/test_import_paths.py -v`
Expected: 1 passed.

- [ ] **Step 5: SMOKE (Isaac machine): run the conversion**

Run: `python setup/import_iiwa.py`
Expected: prints `[OK] wrote USD: ...assets/usd/iiwa7_r800.usd` and the file exists.

- [ ] **Step 6: Commit**

```bash
git add setup/import_iiwa.py docs/assets.md tests/test_import_paths.py
git commit -m "feat: add URDF->USD import for iiwa 7 R800"
```

---

### Task 9: Sim bootstrap

**Files:**
- Create: `src/kuka_sim/sim_app.py`

**Interfaces:**
- Consumes: `isaaclab.app.AppLauncher`.
- Produces: `launch(headless=True, enable_cameras=True, dt=1/120) -> (simulation_app, sim)` where `sim` is a `SimulationContext`.

**Verification: smoke run on the Isaac machine.** No pytest (needs Isaac).

- [ ] **Step 1: Implement `sim_app.py`**

```python
"""Boot Isaac Sim and return (app, SimulationContext)."""

def launch(headless=True, enable_cameras=True, dt=1.0 / 120.0, device="cuda:0"):
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=headless, enable_cameras=enable_cameras)
    simulation_app = app_launcher.app

    # Isaac modules must be imported AFTER the launcher starts.
    from isaaclab.sim import SimulationContext, SimulationCfg
    sim = SimulationContext(SimulationCfg(dt=dt, device=device))
    sim.set_camera_view([2.5, 2.5, 2.0], [0.4, 0.0, 0.3])
    return simulation_app, sim
```

- [ ] **Step 2: SMOKE (Isaac machine): boot and close**

Create a throwaway check (do not commit):
```python
from kuka_sim.sim_app import launch
app, sim = launch(headless=True)
sim.reset()
for _ in range(10):
    sim.step()
print("[OK] sim booted and stepped")
app.close()
```
Run: `python that_check.py`
Expected: `[OK] sim booted and stepped`, clean exit.

- [ ] **Step 3: Commit**

```bash
git add src/kuka_sim/sim_app.py
git commit -m "feat: add Isaac Sim bootstrap (launch)"
```

---

### Task 10: `IiwaArm` — torque-controlled articulation + force sensing

**Files:**
- Create: `src/kuka_sim/robot.py`

**Interfaces:**
- Consumes: `sim_app.launch` (Task 9); Isaac Lab `Articulation`, `ImplicitActuatorCfg`, `ContactSensor`.
- Produces: `IiwaArm` implementing `ArmInterface` (Task 3) + `get_joint_positions()`, `get_ee_force()`, `write()`, `update()`, `reset()`, `initialize()`.

**Verification: smoke run on the Isaac machine.**

- [ ] **Step 1: Implement `robot.py`**

```python
"""Torque-controlled iiwa 7 R800 wrapper implementing ArmInterface."""
import numpy as np


class IiwaArm:
    """Wraps an Isaac Lab Articulation. Direct-torque control (implicit actuator
    with stiffness=damping=0) + a ContactSensor on the flange for contact force.
    """

    N_JOINTS = 7

    def __init__(self, usd_path, prim_path="/World/Robot", ee_body="tool0",
                 surface_prim="/World/Surface", device="cuda:0", dt=1.0 / 120.0):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        self.device = device
        self.dt = dt
        self.ee_body = ee_body
        self._ee_idx = None

        robot_cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0, damping=0.0,      # pass-through: raw torque control
                    effort_limit=320.0, velocity_limit=100.0),
            },
        )
        self.robot = Articulation(robot_cfg)

        contact_cfg = ContactSensorCfg(
            prim_path=f"{prim_path}/{ee_body}",
            update_period=0.0, history_length=4, track_pose=True,   # history → smoothing
            filter_prim_paths_expr=[surface_prim],
        )
        self.contact = ContactSensor(contact_cfg)

    # --- lifecycle ---
    def initialize(self):
        """Call once after sim.reset() so the physics view exists."""
        self._ee_idx = self.robot.find_bodies(self.ee_body)[0][0]

    def write(self):
        self.robot.write_data_to_sim()

    def update(self):
        self.robot.update(self.dt)
        self.contact.update(self.dt)

    def reset(self):
        self.robot.reset()

    # --- ArmInterface ---
    def get_joint_velocities(self):
        return self.robot.data.joint_vel[0].detach().cpu().numpy()

    def get_ee_pose(self):
        pos = self.robot.data.body_pos_w[0, self._ee_idx].detach().cpu().numpy()
        quat = self.robot.data.body_quat_w[0, self._ee_idx].detach().cpu().numpy()
        return pos, quat

    def get_jacobian(self):
        # World-frame Jacobian. Fixed base at world origin (identity root orientation)
        # → world frame == base frame, and pose error is also computed in world frame
        # (body_pos_w / body_quat_w), so the whole control law is frame-consistent.
        # (NVIDIA's OSC tutorial rotates J into the base frame; unnecessary here.)
        jac_body_idx = self._ee_idx - 1        # fixed base: root excluded from jacobians
        J = self.robot.root_physx_view.get_jacobians()[0, jac_body_idx, :, :self.N_JOINTS]
        return J.detach().cpu().numpy()

    def get_gravity_torque(self):
        g = self.robot.root_physx_view.get_gravity_compensation_forces()[0, :self.N_JOINTS]
        return g.detach().cpu().numpy()

    def set_joint_efforts(self, tau):
        import torch
        tau = np.asarray(tau, float)
        if tau.shape != (self.N_JOINTS,):
            raise ValueError(f"tau must be shape (7,), got {tau.shape}")
        t = torch.as_tensor(tau, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.robot.set_joint_effort_target(t)

    # --- extras (demos) ---
    def get_joint_positions(self):
        return self.robot.data.joint_pos[0].detach().cpu().numpy()

    def get_ee_force(self):
        """Contact force (N, world frame) on the flange, smoothed over the sensor
        history. Mirrors NVIDIA's OSC tutorial: mean over history, max over bodies."""
        import torch
        hist = self.contact.data.net_forces_w_history      # (1, T, B, 3)
        mean_over_time = torch.mean(hist, dim=1)            # (1, B, 3)
        f, _ = torch.max(mean_over_time, dim=1)            # (1, 3) strongest-contact body
        return f[0].detach().cpu().numpy()
```

> Confirmed against NVIDIA's OSC tutorial (context7): `get_jacobians()[:, ee_idx-1, :, joint_ids]`, `get_gravity_compensation_forces()[:, joint_ids]`, and the `net_forces_w_history` mean/max force pattern are all verbatim from that reference. **Still confirm at execution:** that `ee_body` matches the actual flange body name in the imported USD — print `robot.data.body_names` in the Step 2 smoke run and set `ee_body` to the real flange name (e.g. `tool0`, `iiwa_link_ee`).

- [ ] **Step 2: SMOKE (Isaac machine): load arm, read state, apply gravity-comp torque**

Throwaway check (do not commit):
```python
from kuka_sim.sim_app import launch
from kuka_sim.robot import IiwaArm
app, sim = launch(headless=True)
arm = IiwaArm(usd_path="assets/usd/iiwa7_r800.usd")
sim.reset(); arm.initialize()
print("bodies:", arm.robot.data.body_names)
for _ in range(200):
    arm.set_joint_efforts(arm.get_gravity_torque())   # hold against gravity
    arm.write(); sim.step(); arm.update()
print("q:", arm.get_joint_positions(), "ee:", arm.get_ee_pose()[0])
app.close()
```
Expected: prints body names (find the flange name), joints stay near home (gravity comp holds it), no crash.

- [ ] **Step 3: Commit**

```bash
git add src/kuka_sim/robot.py
git commit -m "feat: add IiwaArm torque control + contact-force sensing"
```

---

### Task 11: `CompliantSurface` + scene assembly

**Files:**
- Create: `src/kuka_sim/surface.py`
- Create: `src/kuka_sim/scene.py`

**Interfaces:**
- Consumes: Isaac Lab spawners; `IiwaArm` (Task 10).
- Produces: `make_surface(prim_path, pos, size, stiffness, damping) -> RigidObject`; `build_scene(sim, usd_path, surface_kwargs=None) -> dict(arm, surface)` that spawns ground, light, arm, and surface and returns handles.

**Verification: smoke run on the Isaac machine.**

- [ ] **Step 1: Implement `surface.py`**

```python
"""A fixed, compliant contact surface (a soft 'table' the tool presses on)."""

def make_surface(prim_path="/World/Surface", pos=(0.55, 0.0, 0.25),
                 size=(0.4, 0.4, 0.04), stiffness=2000.0, damping=50.0):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    cfg = RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8,
                compliant_contact_stiffness=stiffness,   # >0 enables compliant contact
                compliant_contact_damping=damping),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.55, 0.55)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos)),
    )
    return RigidObject(cfg)
```

> `kinematic_enabled=True` makes the surface immovable (a table). `compliant_contact_stiffness > 0` is what makes contact soft — the swap point for a future `DeformableSurface`.

- [ ] **Step 2: Implement `scene.py`**

```python
"""Assemble the demo scene: ground, light, arm, compliant surface."""
from kuka_sim.robot import IiwaArm
from kuka_sim.surface import make_surface

SURFACE_PRIM = "/World/Surface"


def build_scene(sim, usd_path, surface_kwargs=None):
    import isaaclab.sim as sim_utils

    # ground + dome light — documented standalone spawn idiom: cfg.func(path, cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    surface = make_surface(prim_path=SURFACE_PRIM, **(surface_kwargs or {}))
    arm = IiwaArm(usd_path=usd_path, surface_prim=SURFACE_PRIM)
    return {"arm": arm, "surface": surface}
```

> `cfg.func(path, cfg)` is the documented standalone spawner call (confirmed via context7). Inside an `InteractiveScene` the equivalent is `AssetBaseCfg(prim_path=..., spawn=sim_utils.GroundPlaneCfg())`; we use the standalone form since this scene is built directly, not via a SceneCfg.

- [ ] **Step 3: SMOKE (Isaac machine): build scene, step, confirm no penetration**

Throwaway check (do not commit):
```python
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
app, sim = launch(headless=True)
handles = build_scene(sim, "assets/usd/iiwa7_r800.usd")
arm = handles["arm"]
sim.reset(); arm.initialize()
for _ in range(100):
    arm.set_joint_efforts(arm.get_gravity_torque()); arm.write(); sim.step(); arm.update()
print("[OK] scene built; ee at", arm.get_ee_pose()[0])
app.close()
```
Expected: `[OK] scene built; ...`, arm above the surface, no crash.

- [ ] **Step 4: Commit**

```bash
git add src/kuka_sim/surface.py src/kuka_sim/scene.py
git commit -m "feat: add compliant surface and scene assembly"
```

---

### Task 12: `SceneCamera` — RGB capture

**Files:**
- Create: `src/kuka_sim/camera.py`

**Interfaces:**
- Consumes: Isaac Lab `Camera`/`CameraCfg`.
- Produces: `SceneCamera(prim_path, pos, target, width, height)` with `.update(dt)` and `.capture() -> np.ndarray (H, W, 3) uint8`.

**Verification: smoke run on the Isaac machine (requires `enable_cameras=True`).**

- [ ] **Step 1: Implement `camera.py`**

```python
"""A fixed scene camera that captures RGB frames headless."""
import numpy as np


class SceneCamera:
    def __init__(self, prim_path="/World/DemoCam", pos=(2.2, 1.6, 1.4),
                 target=(0.55, 0.0, 0.3), width=960, height=540, dt=1.0 / 120.0):
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        self.dt = dt
        cfg = CameraCfg(
            prim_path=prim_path,
            update_period=0.0, height=height, width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                             clipping_range=(0.05, 20.0)),
        )
        self.camera = Camera(cfg)
        self._pos = np.array(pos, dtype=float)
        self._target = np.array(target, dtype=float)

    def initialize(self):
        # aim the camera after the physics view exists
        import torch
        self.camera.set_world_poses_from_view(
            torch.tensor([self._pos], dtype=torch.float32),
            torch.tensor([self._target], dtype=torch.float32))

    def update(self):
        self.camera.update(self.dt)

    def capture(self):
        rgb = self.camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:    # drop alpha if RGBA
            rgb = rgb[..., :3]
        return rgb
```

> Flagged: `set_world_poses_from_view` signature and the `rgb` channel count (3 vs 4) need runtime confirmation per the API research. The `[..., :3]` guard already handles RGBA.

- [ ] **Step 2: SMOKE (Isaac machine): capture one frame**

Throwaway check (do not commit):
```python
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.camera import SceneCamera
app, sim = launch(headless=True, enable_cameras=True)
h = build_scene(sim, "assets/usd/iiwa7_r800.usd")
cam = SceneCamera()
sim.reset(); h["arm"].initialize(); cam.initialize()
for _ in range(5):
    sim.step(); h["arm"].update(); cam.update()
frame = cam.capture()
print("[OK] frame:", frame.shape, frame.dtype)
app.close()
```
Expected: `[OK] frame: (540, 960, 3) uint8`.

- [ ] **Step 3: Commit**

```bash
git add src/kuka_sim/camera.py
git commit -m "feat: add SceneCamera RGB capture"
```

---

### Task 13: Free-space impedance demo (`run_demo.py`)

**Files:**
- Create: `scripts/run_demo.py`

**Interfaces:**
- Consumes: `launch`, `build_scene`, `CartesianImpedanceController`.
- Produces: a runnable demo (Deliverable C) — the arm moves to and holds a Cartesian pose compliantly via joint torques.

**Verification: smoke run on the Isaac machine.**

- [ ] **Step 1: Implement `scripts/run_demo.py`**

```python
"""Deliverable C: hold a Cartesian pose with the impedance controller (free space)."""
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.controller import CartesianImpedanceController

USD = "assets/usd/iiwa7_r800.usd"


def main():
    app, sim = launch(headless=True)
    handles = build_scene(sim, USD)
    arm = handles["arm"]
    sim.reset(); arm.initialize()

    ctrl = CartesianImpedanceController(
        arm,
        stiffness=np.array([300, 300, 300, 30, 30, 30.0]),
        damping=np.array([30, 30, 30, 6, 6, 6.0]),
    )
    # target: a pose above the surface, tool pointing down (approx). Read current
    # orientation as the hold orientation to keep the demo robust.
    _, quat0 = arm.get_ee_pose()
    target_pos = np.array([0.55, 0.0, 0.5])
    ctrl.set_target_pose(target_pos, quat0)

    for step in range(1500):
        ctrl.apply(); arm.write(); sim.step(); arm.update()
        if step % 300 == 0:
            err = np.linalg.norm(target_pos - arm.get_ee_pose()[0])
            print(f"step {step}: position error = {err:.4f} m")

    final_err = np.linalg.norm(target_pos - arm.get_ee_pose()[0])
    print(f"[RESULT] final position error = {final_err:.4f} m")
    app.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: SMOKE (Isaac machine): run it**

Run: `python scripts/run_demo.py`
Expected: printed position error **decreases** across steps and the final error is small (tune gains if it does not converge; raise stiffness / add damping if it oscillates).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_demo.py
git commit -m "feat: add free-space impedance hold demo"
```

---

### Task 14: Force-regulation demo (`force_demo.py`)

**Files:**
- Create: `scripts/force_demo.py`

**Interfaces:**
- Consumes: `launch`, `build_scene`, `CartesianImpedanceController`, `ForceLog`.
- Produces: Deliverable D — arm approaches the surface and holds a target normal force; returns the `ForceLog` for reuse by Task 15's recorder.

**Verification: smoke run on the Isaac machine.**

- [ ] **Step 1: Implement `scripts/force_demo.py`**

```python
"""Deliverable D: press the compliant surface and hold a target contact force."""
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.controller import CartesianImpedanceController
from kuka_sim.logging_utils import ForceLog

USD = "assets/usd/iiwa7_r800.usd"
TARGET_FORCE_N = 10.0          # desired downward (-z) contact force


def run(sim=None, app=None, handles=None, target_force=TARGET_FORCE_N,
        steps=2500, on_step=None):
    """Reusable core. If sim/app/handles are None, this owns the sim lifecycle.
    on_step(i) is called each step (Task 15 uses it to grab camera frames)."""
    owns = sim is None
    if owns:
        app, sim = launch(headless=True, enable_cameras=True)
        handles = build_scene(sim, USD)
        sim.reset(); handles["arm"].initialize()
    arm = handles["arm"]

    ctrl = CartesianImpedanceController(
        arm,
        stiffness=np.array([300, 300, 300, 30, 30, 30.0]),
        damping=np.array([30, 30, 30, 6, 6, 6.0]),
    )
    _, quat0 = arm.get_ee_pose()

    # Above the surface (surface top ≈ z=0.25+0.02). Move down onto it, then
    # switch the z task-axis to force control at the target.
    contact_pos = np.array([0.55, 0.0, 0.24])   # slightly into the surface
    ctrl.set_target_pose(contact_pos, quat0)
    z_force_sel = np.array([False, False, True, False, False, False])
    ctrl.set_target_wrench(np.array([0, 0, -target_force, 0, 0, 0.0]), z_force_sel)

    log = ForceLog()
    for i in range(steps):
        ctrl.apply(); arm.write(); sim.step(); arm.update()
        fz = float(arm.get_ee_force()[2])
        log.append(i * arm.dt, arm.get_ee_force())
        if on_step is not None:
            on_step(i)
        if i % 500 == 0:
            print(f"step {i}: |Fz| = {abs(fz):.2f} N (target {target_force:.1f})")

    _, f = log.as_arrays()
    tail = np.linalg.norm(f[-200:], axis=1).mean() if len(f) >= 200 else np.nan
    print(f"[RESULT] mean |F| over last 200 steps = {tail:.2f} N (target {target_force:.1f})")
    if owns:
        app.close()
    return log


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: SMOKE (Isaac machine): run it**

Run: `python scripts/force_demo.py`
Expected: the printed `|Fz|` **rises from 0 and settles near the target** (10 N); the final mean is within tolerance (tune stiffness / surface `compliant_contact_stiffness` / target depth if it overshoots or never contacts).

- [ ] **Step 3: Commit**

```bash
git add scripts/force_demo.py
git commit -m "feat: add force-regulation demo on compliant surface"
```

---

### Task 15: Recorded video + force plot, experiment template, docs

**Files:**
- Create: `scripts/record_demo.py`
- Create: `experiments/bhodi_template.py`
- Create: `docs/quickstart.md`

**Interfaces:**
- Consumes: `force_demo.run`, `SceneCamera`, `frames_to_mp4`, `ForceLog.save_plot`.
- Produces: Deliverables E and F — `out/iiwa_force_demo.mp4` + `out/force_trace.png`, a copy-me template, and a quickstart doc.

**Verification: smoke run on the Isaac machine + pass the full pytest suite.**

- [ ] **Step 1: Implement `scripts/record_demo.py`**

```python
"""Deliverable E: run the force demo, record an MP4 + a force-vs-time plot."""
import os
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.camera import SceneCamera
from kuka_sim.recording import frames_to_mp4
from scripts.force_demo import run, TARGET_FORCE_N, USD

OUT_DIR = "out"
CAPTURE_EVERY = 4          # frames: 120 Hz sim / 4 → 30 fps video


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    app, sim = launch(headless=True, enable_cameras=True)
    handles = build_scene(sim, USD)
    cam = SceneCamera()
    sim.reset(); handles["arm"].initialize(); cam.initialize()

    frames = []

    def on_step(i):
        cam.update()
        if i % CAPTURE_EVERY == 0:
            frames.append(cam.capture())

    log = run(sim=sim, app=app, handles=handles,
              target_force=TARGET_FORCE_N, steps=2500, on_step=on_step)

    mp4 = os.path.join(OUT_DIR, "iiwa_force_demo.mp4")
    png = os.path.join(OUT_DIR, "force_trace.png")
    frames_to_mp4(frames, mp4, fps=30)
    log.save_plot(png, target=TARGET_FORCE_N)
    print(f"[OK] wrote {mp4} ({len(frames)} frames) and {png}")
    app.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement `experiments/bhodi_template.py`**

```python
"""Starting point for Bhodi's force-controlled experiments.

Copy this file, then put your logic in `experiment_step`. You command a
Cartesian pose and/or a contact force and read back the measured force —
the same shape of control you use on the real iiwa.

Run:  python experiments/bhodi_template.py
"""
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.controller import CartesianImpedanceController

USD = "assets/usd/iiwa7_r800.usd"


def experiment_step(i, arm, ctrl):
    """<-- YOUR EXPERIMENT GOES HERE. Called once per sim tick."""
    # Example: hold 8 N normal force on the surface.
    if i == 0:
        _, quat0 = arm.get_ee_pose()
        ctrl.set_target_pose(np.array([0.55, 0.0, 0.24]), quat0)
        ctrl.set_target_wrench(np.array([0, 0, -8.0, 0, 0, 0]),
                               np.array([False, False, True, False, False, False]))
    ctrl.apply()
    if i % 250 == 0:
        print(f"step {i}: measured force = {arm.get_ee_force()} N")


def main():
    app, sim = launch(headless=True)
    handles = build_scene(sim, USD)
    arm = handles["arm"]
    sim.reset(); arm.initialize()
    ctrl = CartesianImpedanceController(
        arm,
        stiffness=np.array([300, 300, 300, 30, 30, 30.0]),
        damping=np.array([30, 30, 30, 6, 6, 6.0]),
    )
    for i in range(2000):
        experiment_step(i, arm, ctrl)
        arm.write(); sim.step(); arm.update()
    app.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `docs/quickstart.md`**

```markdown
# Quickstart

## 1. Install (once, native Windows 11, RTX GPU, recent NVIDIA driver)
    powershell -ExecutionPolicy Bypass -File setup/install.ps1
Then activate the env in each new shell:
    .\env_isaaclab\Scripts\Activate.ps1

## 2. Get the robot description + build the USD (once)
    git clone --depth 1 https://github.com/lbr-stack/lbr_iiwa7_r800_description \
        assets/urdf/lbr_iiwa7_r800_description
    # expand xacro to assets/urdf/iiwa7_r800.urdf if needed (see docs/assets.md)
    python setup/import_iiwa.py        # -> assets/usd/iiwa7_r800.usd

## 3. Run the demos
    python scripts/run_demo.py         # arm holds a pose (free space)
    python scripts/force_demo.py       # arm holds a target contact force
    python scripts/record_demo.py      # -> out/iiwa_force_demo.mp4 + out/force_trace.png

## 4. Write your experiment
Copy `experiments/bhodi_template.py` and edit `experiment_step`. You command a
pose (`ctrl.set_target_pose`) and/or a contact force (`ctrl.set_target_wrench`)
and read force with `arm.get_ee_force()`.

## Control cheatsheet
- `arm.get_joint_positions() / get_joint_velocities()` — joint state
- `arm.get_ee_pose()` — (pos, quat wxyz) of the flange
- `arm.get_ee_force()` — 3-D contact force (N, world frame)
- `ctrl.set_target_pose(pos, quat)` — compliant pose target
- `ctrl.set_target_wrench(wrench6, selection6)` — force-control selected axes

## Notes
- Position control default; torque limits are enforced automatically.
- The surface is a compliant contact (tune `compliant_contact_stiffness` in
  `kuka_sim/surface.py`). It can later be swapped for an FEM deformable body.
- Cutting/material removal is NOT modeled.
```

- [ ] **Step 4: Run the full unit-test suite — expect PASS**

Run: `python -m pytest -v`
Expected: all pure-logic tests pass (Tasks 1–8 tests); `imageio_ffmpeg`-gated tests pass or skip.

- [ ] **Step 5: SMOKE (Isaac machine): record the video**

Run: `python scripts/record_demo.py`
Expected: `[OK] wrote out/iiwa_force_demo.mp4 (...) and out/force_trace.png`; the MP4 plays and shows the arm pressing the surface; the plot shows force settling near 10 N.

- [ ] **Step 6: Commit**

```bash
git add scripts/record_demo.py experiments/bhodi_template.py docs/quickstart.md
git commit -m "feat: add recorded force demo, experiment template, quickstart"
```

---

## Self-Review

**Spec coverage:**
- Turnkey install (Deliverable A) → Task 7 (+ Task 1 packaging). ✓
- Arm loads (B) → Tasks 8 (import) + 10 (load) + 11 (scene). ✓
- Torque control + Cartesian impedance (spec core, Deliverable C) → Tasks 4 (controller) + 10 (effort control) + 13 (demo). ✓
- Force regulation on compliant surface (D) → Tasks 11 (surface) + 14 (demo). ✓
- Rendered video + force plot (E) → Tasks 5 (plot) + 6 (mp4) + 12 (camera) + 15 (record). ✓
- Experiment template + docs (F) → Task 15. ✓
- EE force sensing → Task 10 `get_ee_force` (3-DOF; spec `get_ee_wrench` amended — noted in Global Constraints). ✓
- Torque safety clamp → Task 4 `clamp_torque` + `IIWA_TORQUE_LIMITS`. ✓
- Swappable surface → Task 11 `make_surface` (compliant-stiffness swap point). ✓
- Error handling (missing USD, bad torque length, missing Isaac) → Task 8 `FileNotFoundError`, Task 10 length guard, Task 7 preflight. ✓

**Placeholder scan:** No `TBD`/`TODO`-as-work. The `experiment_step` "YOUR EXPERIMENT GOES HERE" is intentional user-fill in a template, with a working example body — not a plan gap. Isaac-API items explicitly flagged for runtime confirmation carry a concrete default plus the correction site.

**Type consistency:** `set_target_wrench(wrench, selection)` — both `(6,)` in Task 4 and Task 14 caller. `get_ee_force() -> (3,)` consistent in Tasks 10, 14, 15. `get_jacobian() -> (6,7)` consistent in interface, FakeArm, IiwaArm, controller. `IIWA_TORQUE_LIMITS` shape `(7,)` consistent. Quaternion `wxyz` consistent throughout. ✓

**Sim-runtime vs unit-test split:** Tasks 1–8 have real pytest (pure logic). Tasks 9–15 sim glue verified by smoke scripts (documented, with expected observations) because they require the GPU + Isaac runtime — an honest, correct split, not a testing gap.
