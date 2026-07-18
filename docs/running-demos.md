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

The arm aims a **flange probe** forward and presses its tip **on** a stiff
workpiece block, holding a **steady contact force** (~30 N).

```powershell
python scripts/force_demo.py
```

Expected: `|Fx|` settles and holds ~30 N (rock-steady on the stiff block):

```
step 1500: |Fx|=30.x N
step 2000: |Fx|=30.x N
[RESULT] mean |F| over last 200 steps ≈ 30 N (nominal 30)
```

Change the held force by the press depth (`PRESS_TARGET_X`):
`0.42 → ~30 N`, `0.44 → ~40 N`, `0.46 → ~53 N`. The block is stiff (8000) so the
tip presses **on** the face with minimal indent; soften it (`stiffness` in the
surface kwargs) to press **into** a malleable surface instead.

## `record_demo.py` — video + force plot

Runs the force demo with the camera rolling and writes an MP4 + a force-vs-time
plot.

```powershell
python scripts/record_demo.py
# -> out/iiwa_force_demo.mp4   (the arm holding the press)
# -> out/force_trace.png       (contact force over time)
```

Both land in `out/` (git-ignored). Adjust the camera in `record_demo.py`
(`SceneCamera(pos=..., target=...)`). A recording of this demo is on the
[Home page](index.md#see-it-run) ([YouTube](https://youtu.be/DEJWunnK_Vg)).

!!! tip "Clean exits"
    The demos call `os._exit(0)` after `app.close()` — Isaac leaves non-daemon
    threads that otherwise hang the process on shutdown.
