# Control API

The two objects you drive the arm with. Both are in `kuka_sim`.

## `IiwaArm` — the arm

Wraps the Isaac Lab articulation with **direct torque control** and **contact
force sensing**. Construct it after launching Isaac; call `initialize()` once
after `sim.reset()`.

```python
from kuka_sim.robot import IiwaArm

arm = IiwaArm(usd_path="assets/usd/iiwa7_r800.usd")
# ... sim.reset() ...
arm.initialize()
```

| Method | Returns / effect |
|---|---|
| `get_joint_positions()` | `(7,)` joint angles (rad) |
| `get_joint_velocities()` | `(7,)` joint velocities |
| `get_ee_pose()` | `(pos (3,), quat (4,) wxyz)` of the flange |
| `get_ee_force()` | `(3,)` contact force on the flange, world frame |
| `get_jacobian()` | `(6, 7)` end-effector Jacobian (base frame) |
| `get_gravity_torque()` | `(7,)` gravity-compensation torque |
| `set_joint_efforts(tau)` | command 7 joint torques (N·m) |
| `write()` / `update()` | push commands / refresh state (call around `sim.step()`) |
| `reset()` | reset the articulation and contact sensor |

The per-tick loop is always: **command → `write()` → `sim.step()` → `update()`**.

## `CartesianImpedanceController` — the control layer

Maps a Cartesian pose/force target to clamped joint torques:
`tau = Jᵀ(K·xₑ − D·ẋ) + gravity`, with force-selected task axes replaced by a
feed-forward wrench. Consumes an `IiwaArm`.

```python
from kuka_sim.controller import CartesianImpedanceController
import numpy as np

ctrl = CartesianImpedanceController(
    arm,
    stiffness=np.array([1000, 1000, 1000, 30, 30, 30.0]),  # x y z, rx ry rz
    damping=np.array([50, 50, 50, 7, 7, 7.0]),
)
ctrl.set_target_pose(pos, quat)   # impedance pose target
ctrl.apply()                      # compute torques and send to the arm
```

| Method | Effect |
|---|---|
| `set_target_pose(pos (3,), quat (4,))` | set the impedance pose target |
| `set_target_wrench(wrench (6,), selection (6,) bool)` | force-control the selected task axes with a feed-forward wrench |
| `compute()` | return the `(7,)` clamped joint torques for this tick |
| `apply()` | `compute()` then `arm.set_joint_efforts(...)` |

Torque is clamped to the iiwa spec `[176, 176, 110, 110, 110, 40, 40]` N·m, so a
bad gain can't launch the arm.

!!! note "Force control in practice"
    `set_target_wrench` exists and is unit-tested, but for a compliant-contact
    press the robust approach is **impedance-based force control** — command a
    calibrated press *depth* into the surface (see
    [How it works](how-it-works.md)). `force_demo.py` uses that.

## Test double

`tests/fakes.py::FakeArm` implements the same interface with no Isaac, so the
controller is unit-tested (`tests/test_controller.py`) without a GPU.
