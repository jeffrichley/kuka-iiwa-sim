# Write an experiment

Start from the template and put your logic in one function.

```powershell
copy experiments\bhodi_template.py experiments\my_experiment.py
python experiments\my_experiment.py
```

## The shape of an experiment

`experiments/bhodi_template.py` owns the sim lifecycle and calls **your**
`experiment_step(i, arm, ctrl)` once per tick (120 Hz). You command the arm and
read its state there:

```python
def experiment_step(i, arm, ctrl):
    if i == 0:
        # Aim the flange probe forward (+90 deg about Y) and press its tip into
        # the workpiece block; the x press depth (0.44) sets the force (~20 N).
        forward = np.array([0.7071, 0.0, 0.7071, 0.0])
        ctrl.set_target_pose(np.array([0.44, 0.0, 0.62]), forward)

    ctrl.apply()                       # (1) compute + send torques

    if i % 250 == 0:                   # (2) read what happened
        f = arm.get_ee_force()
        print(f"step {i}: |F| = {np.linalg.norm(f):.2f} N")
```

The driver wraps each call with `arm.write() → sim.step() → arm.update()`, so you
never touch the sim loop.

## Common things you'll want

**Hold a target contact force.** Set the press depth from the calibration
(`x=0.44 → ~20 N`, deeper → more). To hold a *different* force, change the x in
`set_target_pose`, or re-run the sweep in `scripts/force_demo.py` for your block
stiffness.

**Change the force mid-run.** Call `ctrl.set_target_pose(...)` with a new x at any
step to ramp or step the press.

**Move along the surface while pressing.** Vary the `y` of the pose target over
time (keep x for the press, z for height) to drag the tool across the panel.

**Read force / position histories.** Use `kuka_sim.logging_utils.ForceLog`:

```python
from kuka_sim.logging_utils import ForceLog
log = ForceLog()
# in the loop:
log.append(i * arm.dt, arm.get_ee_force())
# after:
log.save_plot("out/my_force.png", target=10.0)
```

**Record a video.** Copy the camera + frame-capture pattern from
`scripts/record_demo.py`.

## Tips

- Keep the press in the arm's **dexterous zone** (forward, mid-height). Pressing
  *down* at reach saturates the weak wrist — see [How it works](how-it-works.md).
- Keep pose targets **reachable** (e.g. `z≈0.62` here) so no axis saturates and
  destabilizes contact.
- Aim the flange forward (`[0.7071, 0, 0.7071, 0]`) so the **probe tip** presses,
  not the side of the wrist.
- End your script with `app.close(); os._exit(0)` to avoid Isaac's shutdown hang.
