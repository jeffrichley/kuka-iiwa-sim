"""EE-space gesture generators. Each returns an (n,3) metre offset on box axes
(x=depth, y=lateral, z=height). Pure numpy; summed + modulated by the
choreographer, then mapped into the box and speed-limited by the backend."""
import numpy as np

TWO_PI = 2.0 * np.pi


def _zeros(n):
    return np.zeros((n, 3), float)


def sway(t, freq, amp):
    t = np.asarray(t, float)
    off = _zeros(len(t))
    off[:, 1] = amp * np.sin(TWO_PI * freq * t)
    return off


def bob(t, freq, amp):
    t = np.asarray(t, float)
    off = _zeros(len(t))
    off[:, 2] = amp * np.sin(TWO_PI * freq * t)
    return off


def reach(t, env, amp):
    t = np.asarray(t, float)
    env = np.asarray(env, float)
    off = _zeros(len(t))
    off[:, 0] = amp * env
    return off


def circle(t, freq, amp):
    t = np.asarray(t, float)
    off = _zeros(len(t))
    off[:, 1] = amp * np.cos(TWO_PI * freq * t)
    off[:, 2] = amp * np.sin(TWO_PI * freq * t)
    return off


def figure_eight(t, freq, amp):
    t = np.asarray(t, float)
    off = _zeros(len(t))
    off[:, 1] = amp * np.sin(TWO_PI * freq * t)
    off[:, 2] = amp * np.sin(2.0 * TWO_PI * freq * t)
    return off


def accent(impulse, decay, amp, axis):
    """Causal exp-decay kick after each impulse: convolve the impulse train with
    amp*exp(-decay*k) for k>=0, then place on `axis`."""
    impulse = np.asarray(impulse, float)
    n = len(impulse)
    k = np.arange(n)
    kernel = amp * np.exp(-decay * k)
    conv = np.convolve(impulse, kernel)[:n]
    off = _zeros(n)
    off[:, axis] = conv
    return off
