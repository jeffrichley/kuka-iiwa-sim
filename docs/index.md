# KUKA LBR iiwa 7 R800 — Simulated Digital Twin

A digital twin of the **KUKA LBR iiwa 7 R800** running in **NVIDIA Isaac Sim**,
driven by a **Cartesian impedance / force-control Python API**. It stands in for
the physical arm so force-controlled experiments can continue while the hardware
is unavailable.

The arm reaches forward, presses a compliant surface, and **holds a steady,
tunable contact force** — the core of a contact / force-regulation experiment.

## See it run

The arm aims a flange probe at a workpiece and holds a steady contact force:

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: 8px;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
    src="https://www.youtube.com/embed/DEJWunnK_Vg"
    title="KUKA LBR iiwa 7 R800 — simulated force control in Isaac Sim"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen></iframe>
</div>

Reproduce it with `python scripts/record_demo.py` (see [Run the demos](running-demos.md)).

!!! success "What works today"
    - The iiwa 7 R800 loads from its real URDF and holds itself under gravity
      (torque control, 0.0 rad drift).
    - A Cartesian impedance controller drives the flange to a pose (free space).
    - The arm presses a workpiece with a flange probe and **holds ~30 N steady**
      (tunable by press depth).
    - The whole run renders to an MP4 with a force-vs-time plot.

!!! tip "Bonus: it dances"
    The same twin also **mimics a human dancer**, joint for joint, in full 3D.
    Two arms perform the entire *Evolution of Dance* —
    [watch it](https://youtu.be/NMuR0vQ6Eag) and see
    [The dance module](dance.md).

## Quick map

| You want to… | Go to |
|---|---|
| Set up Isaac Sim + this project | [Install](install.md) |
| Turn the KUKA description into a USD | [Build the robot asset](build-asset.md) |
| Run the demos (pose hold, force press, video) | [Run the demos](running-demos.md) |
| Drive the arm from your own code | [Control API](api.md) |
| Write your own force experiment | [Write an experiment](writing-experiments.md) |
| Understand the control + design choices | [How it works](how-it-works.md) |
| Make the arm **dance** (mimic a real dancer) | [The dance module](dance.md) |

## The stack

| Component | Choice |
|---|---|
| Simulator | Isaac Sim **5.1** |
| Robotics API | Isaac Lab **2.3** |
| Python | **3.11** |
| Robot description | `lbr-stack/lbr_iiwa7_r800_description` |
| OS | Windows 11 (native), RTX GPU |

## Layout

```
src/kuka_sim/     the package — arm, controller, surface, scene, camera, utils
scripts/          runnable demos: run_demo, force_demo, record_demo
experiments/      bhodi_template.py — copy-me starting point
setup/            install + asset-build scripts
docs/             this site
```
