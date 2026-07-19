from kuka_sim.dance.recorder import build_ffmpeg_cmd

def test_none_when_no_extras():
    assert build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4") is None

def test_audio_only():
    c = build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4", audio_path="s.mp3")
    assert "s.mp3" in c and "aac" in c and "-shortest" in c
    assert "-filter_complex" not in c and c[-1] == "o.mp4"

def test_pip_only():
    c = build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4", pip_video_path="d.mp4")
    fc = c[c.index("-filter_complex") + 1]
    assert "scale=-2:" in fc and "overlay=W-w-20:H-h-20" in fc
    assert "[v]" in c and c[-1] == "o.mp4"


def test_pip_height_is_fraction_of_base_height():
    # PiP height must be a fraction of the OUTPUT frame height, not the source's
    # own width (the old bug ballooned a tall vertical clip past the frame).
    c = build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4", pip_video_path="d.mp4",
                         pip_scale=0.25, base_h=540)
    fc = c[c.index("-filter_complex") + 1]
    assert "scale=-2:134" in fc          # int(540*0.25)=135 -> even 134
    assert "iw" not in fc                 # must NOT scale by the source's own width

def test_audio_and_pip_input_order_and_maps():
    c = build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4",
                         audio_path="s.mp3", pip_video_path="d.mp4")
    assert "-filter_complex" in c and "2:a" in c and "aac" in c
    assert c.index("d.mp4") < c.index("s.mp3")   # inputs: base, pip, audio
