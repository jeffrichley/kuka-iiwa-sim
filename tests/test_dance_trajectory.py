import numpy as np
from kuka_sim.dance.trajectory import DanceTrajectory, map_to_workspace, FORWARD_QUAT

def test_forward_quat_is_unit_wxyz():
    assert FORWARD_QUAT.shape == (4,)
    assert np.isclose(np.linalg.norm(FORWARD_QUAT), 1.0)

def test_trajectory_len():
    t = DanceTrajectory(ee_pos=np.zeros((10, 3)))
    assert len(t) == 10 and t.dt == 1.0 / 120.0

def test_map_to_workspace_center_and_corners():
    center = np.array([0.5, 0.0, 0.6]); he = np.array([0.1, 0.15, 0.1])
    assert np.allclose(map_to_workspace([[0, 0, 0]], center, he), [center])
    assert np.allclose(map_to_workspace([[1, 1, 1]], center, he), [center + he])
    assert np.allclose(map_to_workspace([[-1, -1, -1]], center, he), [center - he])

def test_map_to_workspace_clips_out_of_range():
    center = np.array([0.5, 0.0, 0.6]); he = np.array([0.1, 0.1, 0.1])
    out = map_to_workspace([[2.0, 0.0, 0.0]], center, he)
    assert np.allclose(out, [center + [he[0], 0, 0]])
