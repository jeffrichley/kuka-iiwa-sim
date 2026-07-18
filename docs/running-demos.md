# Run the demos

Three runnable scripts, from the repo root with the environment active. Each
launches Isaac headless.

## `run_demo.py` — free-space pose hold

The Cartesian impedance controller drives the flange to a target pose and holds
it (no contact). A quick sanity check that the controller works on the arm.

```powershell
python scripts/run_demo.py
```

Expected: printed position error falls from ~0.9 m to a few cm and holds.

## `force_demo.py` — hold a contact force

The arm aims a **flange probe** forward and presses its tip into a compliant
workpiece block, holding a **steady contact force** (~20 N).

```powershell
python scripts/force_demo.py
```

Expected: after a contact transient, `|Fx|` settles and holds ~20 N:

```
step 1500: |Fx|=24.1 N
step 2000: |Fx|=22.3 N
[RESULT] mean |F| over last 200 steps = 20.8 N (nominal 20)
```

Change the held force by the press depth (`PRESS_TARGET_X` in the script):
`0.42 → ~16 N`, `0.44 → ~20 N`, `0.46 → ~34 N`. A rigid point-tool can't hold much
below ~15 N (it loses contact) — soften the block (`stiffness` in the surface
kwargs) for a lighter, steadier press.

## `record_demo.py` — video + force plot

Runs the force demo with the camera rolling and writes an MP4 + a force-vs-time
plot.

```powershell
python scripts/record_demo.py
# -> out/iiwa_force_demo.mp4   (the arm holding the press)
# -> out/force_trace.png       (contact force over time)
```

Both land in `out/` (git-ignored). Adjust the camera in `record_demo.py`
(`SceneCamera(pos=..., target=...)`).

!!! tip "Clean exits"
    The demos call `os._exit(0)` after `app.close()` — Isaac leaves non-daemon
    threads that otherwise hang the process on shutdown.
