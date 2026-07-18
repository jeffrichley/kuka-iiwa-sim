"""Deliverable D: press the workpiece with the flange PROBE at a steady force.

The arm reaches forward with a probe/tool on its flange (added in the scene) and
presses the probe TIP horizontally (+x) into a compliant workpiece block, holding
a steady contact force. Working in the dexterous zone (forward, mid-height) uses
the strong proximal joints — pressing down at reach saturates the weak 40 N·m
wrist. The flange is aimed forward so the tool tip (not the wrist) does the work.

Force control is IMPEDANCE-BASED: the arm commands a fixed press depth into the
compliant block; the steady force scales with depth. Calibrated by sweep (soft
block, stiffness 400):

    press target x   steady |F|
        0.42            ~16 N
        0.44            ~17 N
        0.46            ~34 N

A rigid point-tool can't hold much below ~15 N (it loses contact), so the demo
holds ~15 N. Change the force with `PRESS_TARGET_X` (and re-sweep if you change
the block stiffness).
"""
import os
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.controller import CartesianImpedanceController
from kuka_sim.logging_utils import ForceLog

USD = "assets/usd/iiwa7_r800.usd"
# Aim the flange +Z (probe/tool axis) toward world +X (at the block): +90° about Y.
FORWARD_QUAT = np.array([0.7071, 0.0, 0.7071, 0.0])
PRESS_TARGET_X = 0.44           # press depth into the block (tune for force)
PRESS_HEIGHT_Z = 0.62           # reachable mid-height (no axis saturation)
NOMINAL_FORCE_N = 20.0          # holds ~20 N (settles from a contact transient)


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
        stiffness=np.array([1000, 1000, 1000, 60, 60, 60.0]),   # firm orientation aim
        damping=np.array([50, 50, 50, 10, 10, 10.0]),
    )
    # Aim the probe at the block and press a calibrated depth into it.
    ctrl.set_target_pose(np.array([press_target_x, 0.0, PRESS_HEIGHT_Z]), FORWARD_QUAT)

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
