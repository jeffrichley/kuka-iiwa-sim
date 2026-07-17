import os
from setup.import_iiwa import resolve_paths

def test_resolve_paths():
    urdf, usd_dir, name = resolve_paths("/repo")
    assert urdf == os.path.join("/repo", "assets/urdf/iiwa7_r800.urdf")
    assert usd_dir == os.path.join("/repo", "assets/usd")
    assert name == "iiwa7_r800.usd"
