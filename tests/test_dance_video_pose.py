import numpy as np
from kuka_sim.dance.video import pose as P
from kuka_sim.dance.video.pose import PoseTrack, _fill_gaps, estimate_poses


def test_fill_gaps_interpolates_interior_nan():
    xy = np.array([[[0.0, 0.0]], [[np.nan, np.nan]], [[2.0, 2.0]]])  # (3,1,2)
    out = _fill_gaps(xy)
    assert np.allclose(out[1, 0], [1.0, 1.0])       # interpolated midpoint
    assert not np.isnan(out).any()


def test_fill_gaps_all_nan_landmark_defaults_center():
    xy = np.full((4, 1, 2), np.nan)
    out = _fill_gaps(xy)
    assert np.allclose(out, 0.5)


def test_estimate_poses_assembles_track(monkeypatch):
    frames = [np.zeros((8, 8, 3), np.uint8)] * 5
    xy = np.full((5, 33, 2), 0.5); xy[2, 10] = [np.nan, np.nan]
    vis = np.ones((5, 33))
    monkeypatch.setattr(P, "_read_frames", lambda path, max_frames=None: (frames, 30.0))
    monkeypatch.setattr(P, "_detect_landmarks", lambda fr: (xy, vis))

    track = estimate_poses("ignored.mp4")
    assert isinstance(track, PoseTrack)
    assert track.xy.shape == (5, 33, 2)
    assert track.fps == 30.0
    assert not np.isnan(track.xy).any()             # gap filled
