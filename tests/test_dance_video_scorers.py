import numpy as np
import pytest
from kuka_sim.dance.video.pose import PoseTrack
from kuka_sim.dance.video.scorers import SCORERS, ScorerCtx


def _track_two_landmarks(F=60, fps=30.0):
    # K=33; landmark 15 (a hand) whips around, everything else still.
    xy = np.full((F, 33, 2), 0.5)
    xy[:, 23] = [0.5, 0.6]; xy[:, 24] = [0.5, 0.6]   # hips (body center)
    xy[:, 11] = [0.4, 0.4]; xy[:, 12] = [0.6, 0.4]   # shoulders
    t = np.linspace(0, 4 * np.pi, F)
    xy[:, 15, 0] = 0.5 + 0.3 * np.sin(t)             # moving, extended hand
    xy[:, 15, 1] = 0.2
    return PoseTrack(xy=xy, visible=np.ones((F, 33)), fps=fps)


def test_all_four_scorers_registered():
    assert set(SCORERS) == {"energy", "music", "saliency", "blend"}


def test_energy_scorer_shape_and_range():
    focus = SCORERS["energy"](_track_two_landmarks(), ScorerCtx())
    assert focus.shape == (60, 2)
    assert focus.min() >= 0.0 and focus.max() <= 1.0


def test_energy_scorer_picks_moving_hand():
    track = _track_two_landmarks()
    focus = SCORERS["energy"](track, ScorerCtx())
    # focus should sit near the hand's y (0.2), not the still body (0.5+)
    assert np.median(focus[:, 1]) < 0.35


def test_music_scorer_requires_beat_env():
    with pytest.raises(ValueError):
        SCORERS["music"](_track_two_landmarks(), ScorerCtx())


def test_music_scorer_runs_with_beat_env():
    track = _track_two_landmarks()
    beat = np.abs(np.sin(np.linspace(0, 8 * np.pi, 60)))
    focus = SCORERS["music"](track, ScorerCtx(beat_env=beat))
    assert focus.shape == (60, 2)


def test_saliency_scorer_requires_frames():
    with pytest.raises(ValueError):
        SCORERS["saliency"](_track_two_landmarks(), ScorerCtx())


def test_blend_scorer_requires_beat_env():
    with pytest.raises(ValueError):
        SCORERS["blend"](_track_two_landmarks(), ScorerCtx())


def test_blend_scorer_runs():
    track = _track_two_landmarks()
    beat = np.abs(np.sin(np.linspace(0, 8 * np.pi, 60)))
    focus = SCORERS["blend"](track, ScorerCtx(beat_env=beat))
    assert focus.shape == (60, 2)
    assert focus.min() >= 0.0 and focus.max() <= 1.0
