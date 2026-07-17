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
