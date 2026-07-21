"""Assemble the dance video: encode frames, mux audio, overlay a corner PiP."""

import os
import subprocess
import tempfile


def build_ffmpeg_cmd(ffmpeg, base_path, out_path, audio_path=None,
                     pip_video_path=None, pip_scale=0.22, margin=20, base_h=540):
    """Build the ffmpeg arg list to mux audio and/or overlay a corner PiP onto the
    base (silent) render. Inputs are ordered base, pip, audio. Returns None when
    neither extra is requested (the caller then uses the base render as the output).

    pip_scale is the PiP's fraction of the OUTPUT FRAME HEIGHT (base_h), so the
    inset stays a small corner regardless of the source clip's own resolution —
    scaling by the source's own width blew a tall vertical clip up past the frame.
    """
    if audio_path is None and pip_video_path is None:
        return None
    cmd = [ffmpeg, "-y", "-i", base_path]
    if pip_video_path is not None:
        cmd += ["-i", pip_video_path]
    if audio_path is not None:
        cmd += ["-i", audio_path]

    if pip_video_path is not None:
        pip_h = int(base_h * pip_scale)
        pip_h -= pip_h % 2                            # even height for h.264/yuv420p
        # width auto (-2) keeps aspect and rounds even; height is a fixed fraction
        # of the frame so the inset is always a small corner.
        overlay = (f"[1:v]scale=-2:{pip_h}[p];"
                   f"[0:v][p]overlay=W-w-{margin}:H-h-{margin}[v]")
        cmd += ["-filter_complex", overlay, "-map", "[v]"]
        if audio_path is not None:                    # audio is input index 2
            cmd += ["-map", "2:a", "-c:a", "aac", "-shortest"]
    else:                                             # audio only, no re-encode of video
        cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest"]

    cmd += [out_path]
    return cmd


def record(frames, out_path, fps=30, audio_path=None, pip_video_path=None):
    """Encode frames to an MP4; if audio and/or a PiP source is given, run ffmpeg
    to mux/overlay them. Returns out_path."""
    import imageio_ffmpeg
    from kuka_sim.recording import frames_to_mp4

    if audio_path is None and pip_video_path is None:
        return frames_to_mp4(frames, out_path, fps=fps)

    import numpy as np
    base_h = int(np.asarray(frames[0]).shape[0]) if len(frames) else 540
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "base.mp4")
        frames_to_mp4(frames, base, fps=fps)
        cmd = build_ffmpeg_cmd(ffmpeg, base, out_path, audio_path, pip_video_path,
                               base_h=base_h)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "ffmpeg failed:\n" + e.stderr.decode(errors="replace")) from e
    return out_path
