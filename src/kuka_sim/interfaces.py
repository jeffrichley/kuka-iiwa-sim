"""The minimal arm contract the controller depends on."""
from typing import Protocol, Tuple
import numpy as np


class ArmInterface(Protocol):
    def get_joint_velocities(self) -> np.ndarray: ...           # (7,)
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]: ...  # (3,), (4,) wxyz
    def get_jacobian(self) -> np.ndarray: ...                   # (6, 7) base frame
    def get_gravity_torque(self) -> np.ndarray: ...             # (7,)
    def set_joint_efforts(self, tau: np.ndarray) -> None: ...   # (7,)
