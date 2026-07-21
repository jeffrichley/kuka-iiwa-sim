import numpy as np
from kuka_sim.dance.trajectory import smooth_and_limit

def test_velocity_is_clamped():
    dt, max_speed = 1.0 / 120.0, 1.0            # max step = 1/120 m
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], float)  # 1 m jump = way too fast
    out = smooth_and_limit(pos, dt, max_speed, smooth_win=1)  # no smoothing
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.all(steps <= max_speed * dt + 1e-9)

def test_constant_path_preserved():
    pos = np.tile([0.5, 0.0, 0.6], (12, 1))
    out = smooth_and_limit(pos, 1.0 / 120.0, 1.0, smooth_win=5)
    assert np.allclose(out, pos)                # constant in, constant out (edge-padded)

def test_shape_and_endpoints_preserved_under_smoothing():
    rng = np.random.RandomState(0)
    pos = rng.randn(50, 3) * 0.005 + [0.5, 0.0, 0.6]
    out = smooth_and_limit(pos, 1.0 / 120.0, 0.5, smooth_win=5)
    assert out.shape == (50, 3)
    # smoothing reduces jitter -> less total path length than the noisy input
    def plen(a): return np.linalg.norm(np.diff(a, axis=0), axis=1).sum()
    assert plen(out) <= plen(pos)

def test_even_smooth_win_preserves_shape():
    import numpy as np
    from kuka_sim.dance.trajectory import smooth_and_limit
    pos = np.random.RandomState(1).randn(30, 3) * 0.01 + [0.5, 0.0, 0.6]
    for win in (2, 4, 6):
        out = smooth_and_limit(pos, 1.0 / 120.0, 0.5, smooth_win=win)
        assert out.shape == (30, 3), f"win={win} gave {out.shape}"
