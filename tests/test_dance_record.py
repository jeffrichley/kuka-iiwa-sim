import numpy as np
import pytest
from kuka_sim.dance.recorder import record

def test_record_plain_writes_mp4(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    frames = [np.full((64, 64, 3), i * 20, dtype=np.uint8) for i in range(8)]
    out = tmp_path / "clip.mp4"
    record(frames, str(out), fps=8)
    assert out.exists() and out.stat().st_size > 0

def test_record_with_audio_muxes(tmp_path):
    import shutil, subprocess, imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # make a 1s silent audio to mux
    wav = tmp_path / "sil.wav"
    subprocess.run([ff, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", "1", str(wav)], check=True, capture_output=True)
    frames = [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(24)]
    out = tmp_path / "with_audio.mp4"
    record(frames, str(out), fps=24, audio_path=str(wav))
    assert out.exists() and out.stat().st_size > 0
