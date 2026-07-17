import numpy as np
from kuka_sim.logging_utils import ForceLog

def test_log_accumulates_and_shapes():
    log = ForceLog()
    log.append(0.0, [0, 0, -1.0])
    log.append(0.1, [0, 0, -4.0])
    t, f = log.as_arrays()
    assert t.shape == (2,) and f.shape == (2, 3)
    assert np.isclose(t[1], 0.1)

def test_save_plot_writes_file(tmp_path):
    log = ForceLog()
    for i in range(5):
        log.append(i * 0.1, [0, 0, -float(i)])
    out = tmp_path / "force.png"
    log.save_plot(str(out), target=5.0)
    assert out.exists() and out.stat().st_size > 0
