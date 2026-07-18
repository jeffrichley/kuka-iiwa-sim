"""Starting point for Bhodi's force-controlled experiments.

Copy this file and put your logic in `experiment_step`. You command the arm's
Cartesian pose via impedance control and read back the contact force — the same
shape of control you'd use on the real iiwa. Pressing a calibrated depth into the
workpiece sets the contact force (see scripts/force_demo.py for the depth -> force
calibration: x=0.42 -> ~30 N, deeper -> more).

Run:  python experiments/bhodi_template.py
"""
import os
import numpy as np
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.controller import CartesianImpedanceController

USD = "assets/usd/iiwa7_r800.usd"


def experiment_step(i, arm, ctrl):
    """<-- YOUR EXPERIMENT GOES HERE. Called once per sim tick (120 Hz).

    Available each step:
      arm.get_joint_positions() / get_joint_velocities()  -> (7,)
      arm.get_ee_pose()   -> (pos (3,), quat (4,) wxyz)
      arm.get_ee_force()  -> (3,) contact force in world frame
      ctrl.set_target_pose(pos, quat)          # impedance pose target
      ctrl.set_target_wrench(wrench6, sel6)    # force-control selected task axes
      ctrl.apply()                             # compute torques + send to the arm
    """
    if i == 0:
        # Example: aim the flange probe forward (+90 deg about Y so the tool axis
        # points at the block) and press its tip on the workpiece. The x press
        # depth (0.42) into the stiff block sets the force (~30 N).
        forward = np.array([0.7071, 0.0, 0.7071, 0.0])
        ctrl.set_target_pose(np.array([0.42, 0.0, 0.62]), forward)

    ctrl.apply()

    if i % 250 == 0:
        f = arm.get_ee_force()
        print(f"step {i}: contact force = {np.round(f, 2)} N  |F| = {np.linalg.norm(f):.2f}",
              flush=True)


def main():
    app, sim = launch(headless=True)
    handles = build_scene(sim, USD)
    arm = handles["arm"]
    sim.reset(); arm.initialize()

    ctrl = CartesianImpedanceController(
        arm,
        stiffness=np.array([1000, 1000, 1000, 60, 60, 60.0]),   # firm orientation aim
        damping=np.array([50, 50, 50, 10, 10, 10.0]),
    )

    for i in range(2000):
        experiment_step(i, arm, ctrl)
        arm.write(); sim.step(); arm.update()

    app.close()
    os._exit(0)   # Isaac leaves non-daemon threads; force a clean exit


if __name__ == "__main__":
    main()
