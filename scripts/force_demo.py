"""Deliverable D: press the compliant panel and hold a steady contact force.

The surface is a VERTICAL panel in front of the arm; the arm reaches forward and
presses HORIZONTALLY (+x) into it, in its dexterous zone (strong proximal joints,
not the weak 40 N·m wrist that saturates when pressing down at reach).

Force control is IMPEDANCE-BASED: the arm commands a fixed press depth into the
compliant panel, and the steady contact force scales monotonically with depth.
This is far more robust here than a feed-forward force command, which is too weak
to hold contact against the arm's pose-restoring dynamics. Calibrated by sweep:

    press target x   steady |F|
        0.58            ~10 N
        0.60            ~19 N
        0.62            ~43 N
        0.70            ~63 N

Change the held force by adjusting `press_target_x`.
"""
import os
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.controller import CartesianImpedanceController
from kuka_sim.logging_utils import ForceLog

USD = "assets/usd/iiwa7_r800.usd"
PRESS_TARGET_X = 0.565           # press depth into the panel (tune for force; ~10 N)
NOMINAL_FORCE_N = 10.0
# z-target at the flange's natural forward-reach height so the z-axis does NOT
# saturate (an unreachable z-target commands permanent max downward torque, which
# drives a slow contact instability). Keeps only the x-press acting on the panel.
PRESS_HEIGHT_Z = 0.80


def run(sim=None, app=None, handles=None, steps=2500, on_step=None,
        press_target_x=PRESS_TARGET_X, enable_cameras_if_owned=False):
    """Reusable core. If sim/app/handles are None, this owns the sim lifecycle.
    on_step(i) is called each step (Task 15 uses it to grab camera frames)."""
    owns = sim is None
    if owns:
        app, sim = launch(headless=True, enable_cameras=enable_cameras_if_owned)
        handles = build_scene(sim, USD)
        sim.reset(); handles["arm"].initialize()
    arm = handles["arm"]

    ctrl = CartesianImpedanceController(
        arm,
        stiffness=np.array([1000, 1000, 1000, 30, 30, 30.0]),
        damping=np.array([50, 50, 50, 7, 7, 7.0]),
    )
    _, quat0 = arm.get_ee_pose()
    # Reach forward and press a fixed depth into the panel; the compliant contact
    # converts that depth into a steady force (~10 N at x=0.58).
    ctrl.set_target_pose(np.array([press_target_x, 0.0, PRESS_HEIGHT_Z]), quat0)

    log = ForceLog()
    for i in range(steps):
        ctrl.apply(); arm.write(); sim.step(); arm.update()
        log.append(i * arm.dt, arm.get_ee_force())
        if on_step is not None:
            on_step(i)
        if i % 500 == 0:
            fx = float(arm.get_ee_force()[0])
            ee = arm.get_ee_pose()[0]
            print(f"step {i}: |Fx|={abs(fx):.2f} N  ee_x={ee[0]:.3f}", flush=True)

    _, f = log.as_arrays()
    tail = np.linalg.norm(f[-200:], axis=1).mean() if len(f) >= 200 else np.nan
    print(f"[RESULT] mean |F| over last 200 steps = {tail:.2f} N "
          f"(nominal {NOMINAL_FORCE_N:.0f})", flush=True)
    if owns:
        app.close()
        os._exit(0)
    return log


if __name__ == "__main__":
    run()
