import numpy as np
from kuka_sim.dance.video.saliency import spectral_residual, _resize, _boxblur


def test_resize_nearest_shape():
    img = np.arange(100, dtype=float).reshape(10, 10)
    out = _resize(img, (5, 5))
    assert out.shape == (5, 5)


def test_boxblur_preserves_shape_and_smooths():
    a = np.zeros((8, 8)); a[4, 4] = 1.0
    out = _boxblur(a, 3)
    assert out.shape == (8, 8)
    assert out[4, 4] < 1.0            # energy spread to neighbors
    assert out[3, 4] > 0.0


def test_spectral_residual_is_unit_range():
    rng = np.random.RandomState(0)
    gray = rng.rand(48, 64)
    sm = spectral_residual(gray, out_hw=(32, 32))
    assert sm.shape == (32, 32)
    assert sm.min() >= 0.0 and sm.max() <= 1.0


def test_spectral_residual_localizes_point_feature():
    # spectral residual peaks AT a localized point feature (a solid region
    # instead rings energy to corners — this is the property the scorer uses).
    gray = np.zeros((64, 64))
    gray[31, 31] = 1.0
    sm = spectral_residual(gray, out_hw=(64, 64))
    ay, ax = np.unravel_index(sm.argmax(), sm.shape)
    assert abs(ay - 31) <= 3 and abs(ax - 31) <= 3
