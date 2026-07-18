# kuka — Simulated KUKA LBR iiwa 7 R800

A digital twin of the KUKA LBR iiwa 7 R800 in NVIDIA Isaac Sim, driven by a
Cartesian impedance / hybrid-force Python API. Built so Bhodi can run his
force-controlled experiments in sim while the physical arm is down.

See `docs/superpowers/plans/2026-07-17-kuka-iiwa-force-sim.md` for the build plan
and `docs/quickstart.md` (added in the final step) for setup and usage.

## Layout
- `src/kuka_sim/` — the package (arm, controller, surface, scene, camera, utils)
- `scripts/` — runnable demos (`run_demo`, `force_demo`, `record_demo`)
- `experiments/` — Bhodi's starting template
- `setup/` — install + URDF-import scripts
