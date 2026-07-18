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
