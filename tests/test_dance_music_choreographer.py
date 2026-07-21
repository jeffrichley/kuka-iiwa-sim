import numpy as np
import pytest
from kuka_sim.dance.music.choreographer import (
    choreograph, euler_to_quat, quat_mul, STYLES,
)
from kuka_sim.dance.music.features import FeatureTrack


def _track(n=600, dt=1/120):
    rng = np.random.RandomState(0)
    beats = np.zeros(n); beats[::60] = 1.0
    onsets = np.zeros(n); onsets[::40] = 1.0
    ramp = np.linspace(0, 1, n)
    return FeatureTrack(
        dt=dt, n=n,
        rms=ramp, bass=np.abs(np.sin(np.linspace(0, 9, n))),
        mid=rng.rand(n), treble=rng.rand(n),
        beats=beats, onsets=onsets, tempo=120.0,
    )


def test_euler_to_quat_is_unit_and_identity_at_zero():
    q = euler_to_quat(0.0, 0.0, 0.0)
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])
    q2 = euler_to_quat(0.1, -0.2, 0.3)
    assert np.isclose(np.linalg.norm(q2), 1.0)


def test_quat_mul_identity():
    q = np.array([0.7071, 0.0, 0.7071, 0.0])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_mul(ident, q), q)


def test_choreograph_positions_inside_box():
    center = np.array([0.5, 0.0, 0.7]); half = np.array([0.10, 0.18, 0.14])
    traj = choreograph(_track(), style="waltz", camera="hero",
                       box_center=tuple(center), box_half_extents=tuple(half))
    assert traj.ee_pos.shape == (600, 3)
    assert np.all(traj.ee_pos <= center + half + 1e-6)
    assert np.all(traj.ee_pos >= center - half - 1e-6)


def test_choreograph_quats_unit_norm():
    traj = choreograph(_track(), style="waltz")
    assert traj.ee_quat.shape == (600, 4)
    norms = np.linalg.norm(traj.ee_quat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_hero_camera_is_none_cinematic_is_track():
    hero = choreograph(_track(), camera="hero")
    assert hero.cam_pos is None and hero.cam_target is None
    cine = choreograph(_track(), camera="cinematic")
    assert cine.cam_pos.shape == (600, 3)
    assert cine.cam_target.shape == (600, 3)


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        choreograph(_track(), style="nope")


def test_styles_have_required_presets():
    assert "waltz" in STYLES and "epic" in STYLES
