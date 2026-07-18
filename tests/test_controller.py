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
    c = _ctrl(arm, d=np.array([2.0, 0, 0, 0, 0, 0]))
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
