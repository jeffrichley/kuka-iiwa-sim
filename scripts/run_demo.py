"""Deliverable C: hold a Cartesian pose with the impedance controller (free space)."""
import os
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
        stiffness=np.array([800, 800, 800, 40, 40, 40.0]),
        damping=np.array([40, 40, 40, 8, 8, 8.0]),
    )
    # target: a pose above the surface, holding the current (home) orientation.
    _, quat0 = arm.get_ee_pose()
    target_pos = np.array([0.55, 0.0, 0.5])
    ctrl.set_target_pose(target_pos, quat0)

    for step in range(1500):
        ctrl.apply(); arm.write(); sim.step(); arm.update()
        if step % 300 == 0:
            err = np.linalg.norm(target_pos - arm.get_ee_pose()[0])
            print(f"step {step}: position error = {err:.4f} m", flush=True)

    final_err = np.linalg.norm(target_pos - arm.get_ee_pose()[0])
    print(f"[RESULT] final position error = {final_err:.4f} m", flush=True)
    app.close()
    os._exit(0)   # Isaac leaves non-daemon threads; force a clean exit


if __name__ == "__main__":
    main()
