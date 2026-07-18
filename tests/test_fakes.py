import numpy as np
from tests.fakes import FakeArm

def test_fakearm_defaults_and_records_efforts():
    arm = FakeArm()
    assert arm.get_jacobian().shape == (6, 7)
    assert arm.get_gravity_torque().shape == (7,)
    pos, quat = arm.get_ee_pose()
    assert pos.shape == (3,) and quat.shape == (4,)
    assert arm.get_joint_velocities().shape == (7,)
    arm.set_joint_efforts(np.arange(7.0))
    assert np.allclose(arm.applied, np.arange(7.0))

def test_fakearm_rejects_bad_effort_length():
    arm = FakeArm()
    try:
        arm.set_joint_efforts(np.zeros(6))
        assert False, "expected ValueError"
    except ValueError:
        pass
