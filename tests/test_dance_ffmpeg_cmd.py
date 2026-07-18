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
    assert "scale=iw*0.25" in fc and "overlay=W-w-20:H-h-20" in fc
    assert "[v]" in c and c[-1] == "o.mp4"

def test_audio_and_pip_input_order_and_maps():
    c = build_ffmpeg_cmd("ffmpeg", "b.mp4", "o.mp4",
                         audio_path="s.mp3", pip_video_path="d.mp4")
    assert "-filter_complex" in c and "2:a" in c and "aac" in c
    assert c.index("d.mp4") < c.index("s.mp3")   # inputs: base, pip, audio
