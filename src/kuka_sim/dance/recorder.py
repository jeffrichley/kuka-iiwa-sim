"""Assemble the dance video: encode frames, mux audio, overlay a corner PiP."""


def build_ffmpeg_cmd(ffmpeg, base_path, out_path, audio_path=None,
                     pip_video_path=None, pip_scale=0.25, margin=20):
    """Build the ffmpeg arg list to mux audio and/or overlay a corner PiP onto the
    base (silent) render. Inputs are ordered base, pip, audio. Returns None when
    neither extra is requested (the caller then uses the base render as the output)."""
    if audio_path is None and pip_video_path is None:
        return None
    cmd = [ffmpeg, "-y", "-i", base_path]
    if pip_video_path is not None:
        cmd += ["-i", pip_video_path]
    if audio_path is not None:
        cmd += ["-i", audio_path]

    if pip_video_path is not None:
        overlay = (f"[1:v]scale=iw*{pip_scale}:-1[p];"
                   f"[0:v][p]overlay=W-w-{margin}:H-h-{margin}[v]")
        cmd += ["-filter_complex", overlay, "-map", "[v]"]
        if audio_path is not None:                    # audio is input index 2
            cmd += ["-map", "2:a", "-c:a", "aac", "-shortest"]
    else:                                             # audio only, no re-encode of video
        cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest"]

    cmd += [out_path]
    return cmd
