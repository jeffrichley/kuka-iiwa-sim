import numpy as np
from kuka_sim.dance.video.retarget import retarget


def test_retarget_center_maps_to_box_center():
    focus = np.full((30, 2), 0.5)
    traj = retarget(focus, fps=30.0, video_path="d.mp4")
    assert np.allclose(traj.ee_pos[:, :], [0.5, 0.0, 0.7], atol=1e-6)


def test_retarget_extremes_stay_in_box():
    center = np.array([0.5, 0.0, 0.7]); half = np.array([0.10, 0.18, 0.14])
    focus = np.array([[1.0, 0.0], [0.0, 1.0]] * 15)   # (30,2)
    traj = retarget(focus, fps=30.0)
    assert np.all(traj.ee_pos <= center + half + 1e-6)
    assert np.all(traj.ee_pos >= center - half - 1e-6)


def test_retarget_maps_axes():
    # image x=1 -> lateral +y (max); image y=0 -> height +z (max)
    focus = np.full((30, 2), 0.5); focus[:, 0] = 1.0; focus[:, 1] = 0.0
    traj = retarget(focus, fps=30.0)
    assert traj.ee_pos[:, 1].mean() > 0.15    # near +y edge (0 + 0.18)
    assert traj.ee_pos[:, 2].mean() > 0.82    # near +z edge (0.7 + 0.14)


def test_retarget_resamples_to_sim_grid():
    focus = np.full((30, 2), 0.5)             # 1s @30fps -> ~120 samples @1/120
    traj = retarget(focus, fps=30.0)
    assert traj.ee_pos.shape[0] == round(30 / 30.0 / (1 / 120.0))
    assert traj.dt == 1 / 120


def test_retarget_sets_pip_and_audio():
    traj = retarget(np.full((10, 2), 0.5), fps=30.0, video_path="dance.mp4")
    assert traj.pip_video_path == "dance.mp4"
    assert traj.audio_path == "dance.mp4"      # defaults to the video's own audio
