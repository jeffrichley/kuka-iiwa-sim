import numpy as np
from kuka_sim.dance.music import moves


def _t(n=240, dt=1/120):
    return np.arange(n) * dt


def test_sway_is_lateral_only_and_bounded():
    off = moves.sway(_t(), freq=1.0, amp=0.05)
    assert off.shape == (240, 3)
    assert np.allclose(off[:, 0], 0.0) and np.allclose(off[:, 2], 0.0)
    assert np.abs(off[:, 1]).max() <= 0.05 + 1e-9


def test_bob_is_vertical_only():
    off = moves.bob(_t(), freq=2.0, amp=0.03)
    assert np.allclose(off[:, 0], 0.0) and np.allclose(off[:, 1], 0.0)
    assert np.abs(off[:, 2]).max() <= 0.03 + 1e-9


def test_reach_follows_envelope_on_x():
    env = np.linspace(0, 1, 240)
    off = moves.reach(_t(), env=env, amp=0.08)
    assert np.allclose(off[:, 1], 0.0) and np.allclose(off[:, 2], 0.0)
    assert np.isclose(off[-1, 0], 0.08)
    assert np.isclose(off[0, 0], 0.0)


def test_circle_traces_yz_circle():
    off = moves.circle(_t(), freq=1.0, amp=0.04)
    r = np.sqrt(off[:, 1] ** 2 + off[:, 2] ** 2)
    assert np.allclose(r, 0.04, atol=1e-9)
    assert np.allclose(off[:, 0], 0.0)


def test_figure_eight_shape():
    off = moves.figure_eight(_t(), freq=1.0, amp=0.04)
    assert off.shape == (240, 3)
    assert np.allclose(off[:, 0], 0.0)
    assert np.abs(off[:, 1]).max() <= 0.04 + 1e-9


def test_accent_decays_after_impulse():
    imp = np.zeros(240); imp[10] = 1.0
    off = moves.accent(imp, decay=0.3, amp=0.05, axis=2)
    assert np.allclose(off[:10, 2], 0.0)          # nothing before the impulse
    assert np.isclose(off[10, 2], 0.05)            # peak at the impulse
    assert off[11, 2] < off[10, 2]                 # decays afterward
    assert np.allclose(off[:, 0], 0.0) and np.allclose(off[:, 1], 0.0)
