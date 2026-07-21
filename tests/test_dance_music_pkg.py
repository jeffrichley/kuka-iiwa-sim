def test_music_package_imports():
    import kuka_sim.dance.music  # noqa: F401


def test_librosa_available():
    import librosa  # noqa: F401
    assert hasattr(librosa, "beat")
