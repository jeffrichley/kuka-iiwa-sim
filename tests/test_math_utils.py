import numpy as np
from kuka_sim.math_utils import quat_conj, quat_mul, orientation_error

def test_quat_mul_identity():
    q = np.array([0.0, 0.0, 1.0, 0.0])          # wxyz
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_mul(ident, q), q)

def test_quat_conj():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(quat_conj(q), [1.0, -2.0, -3.0, -4.0])

def test_orientation_error_zero_when_equal():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(orientation_error(q, q), np.zeros(3))

def test_orientation_error_90deg_about_z():
    c = np.cos(np.pi / 4); s = np.sin(np.pi / 4)
    q_des = np.array([c, 0.0, 0.0, s])           # +90° about z
    q_cur = np.array([1.0, 0.0, 0.0, 0.0])
    err = orientation_error(q_des, q_cur)
    assert np.allclose(err, [0.0, 0.0, np.pi / 2], atol=1e-6)

def test_quat_mul_nonidentity_hamilton_relations():
    i = np.array([0.0, 1.0, 0.0, 0.0])
    j = np.array([0.0, 0.0, 1.0, 0.0])
    k = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(quat_mul(i, j), k)         # i*j = k
    assert np.allclose(quat_mul(j, i), -k)        # j*i = -k
    assert np.allclose(quat_mul(j, k), i)         # j*k = i
    assert np.allclose(quat_mul(k, i), j)         # k*i = j

def test_orientation_error_antipodal_is_zero():
    # q and -q represent the SAME orientation; error must be ~0, and this
    # exercises the shortest-path sign flip (q_err[0] < 0).
    q_cur = np.array([1.0, 0.0, 0.0, 0.0])
    q_des = -q_cur
    assert np.allclose(orientation_error(q_des, q_cur), np.zeros(3), atol=1e-6)

def test_orientation_error_takes_short_way_past_180deg():
    # +270° about z is the same as -90°; the shortest-path result is [0,0,-pi/2].
    c = np.cos(np.deg2rad(135.0)); s = np.sin(np.deg2rad(135.0))
    q_des = np.array([c, 0.0, 0.0, s])            # q_err[0] will be < 0 before the flip
    q_cur = np.array([1.0, 0.0, 0.0, 0.0])
    err = orientation_error(q_des, q_cur)
    assert np.allclose(err, [0.0, 0.0, -np.pi / 2], atol=1e-6)
