# KUKA LBR iiwa 7 R800 — Simulated Digital Twin

A digital twin of the **KUKA LBR iiwa 7 R800** running in **NVIDIA Isaac Sim**,
driven by a **Cartesian impedance / force-control Python API**. It stands in for
the physical arm so force-controlled experiments can continue while the hardware
is unavailable.

The arm reaches forward, presses a compliant surface, and **holds a steady,
tunable contact force** — the core of a contact / force-regulation experiment.

!!! success "What works today"
    - The iiwa 7 R800 loads from its real URDF and holds itself under gravity
      (torque control, 0.0 rad drift).
    - A Cartesian impedance controller drives the flange to a pose (free space).
    - The arm presses a compliant panel and **holds ~10 N steady** (tunable by
      press depth).
    - The whole run renders to an MP4 with a force-vs-time plot.

## Quick map

| You want to… | Go to |
|---|---|
| Set up Isaac Sim + this project | [Install](install.md) |
| Turn the KUKA description into a USD | [Build the robot asset](build-asset.md) |
| Run the demos (pose hold, force press, video) | [Run the demos](running-demos.md) |
| Drive the arm from your own code | [Control API](api.md) |
| Write your own force experiment | [Write an experiment](writing-experiments.md) |
| Understand the control + design choices | [How it works](how-it-works.md) |

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
