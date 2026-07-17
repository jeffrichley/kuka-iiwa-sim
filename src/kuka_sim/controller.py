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
