"""Contact-force logging and a force-vs-time plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless-safe; no display needed
import matplotlib.pyplot as plt


class ForceLog:
    def __init__(self):
        self._t = []
        self._f = []

    def append(self, t, force):
        self._t.append(float(t))
        self._f.append(np.asarray(force, float).reshape(3).copy())

    def as_arrays(self):
        return np.asarray(self._t), np.asarray(self._f)

    def save_plot(self, path, target=None):
        t, f = self.as_arrays()
        mag = np.linalg.norm(f, axis=1) if f.ndim == 2 else f
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, mag, label="|F| measured")
        if target is not None:
            ax.axhline(target, ls="--", color="r", label="target")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("contact force (N)")
        ax.set_title("End-effector contact force")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
