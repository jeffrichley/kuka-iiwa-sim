import numpy as np
import pytest
from kuka_sim.recording import frames_to_mp4

def test_frames_to_mp4_writes_file(tmp_path):
    pytest.importorskip("imageio_ffmpeg")   # skip if the ffmpeg backend is absent
    frames = [np.full((64, 64, 3), i * 20, dtype=np.uint8) for i in range(10)]
    out = tmp_path / "clip.mp4"
    frames_to_mp4(frames, str(out), fps=10)
    assert out.exists() and out.stat().st_size > 0

def test_frames_to_mp4_handles_rgba_and_float(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    frames = [np.ones((32, 32, 4), dtype=np.float32) * 0.5 for _ in range(5)]
    out = tmp_path / "clip2.mp4"
    frames_to_mp4(frames, str(out), fps=5)
    assert out.exists() and out.stat().st_size > 0
