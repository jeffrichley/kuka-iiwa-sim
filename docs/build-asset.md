# Build the robot asset

Isaac needs the KUKA arm as a **USD**. We generate it from the real
`lbr-stack/lbr_iiwa7_r800_description` in three steps. Everything under
`assets/` is git-ignored (fetched / generated), so this runs once per machine.

## 1. Fetch the description

```bash
git clone --depth 1 https://github.com/lbr-stack/lbr_iiwa7_r800_description \
    assets/urdf/lbr_iiwa7_r800_description
```

## 2. Expand the xacro → plain URDF

The description ships as **xacro** (a macro that loads `config/joint_limits.yaml`),
which Isaac's importer can't read directly. `setup/expand_xacro.py` expands it
**without a ROS install** — it substitutes `$(find …)` package refs with absolute
paths and rewrites `package://` mesh refs to be relative.

It needs the `xacro` pip package (not Isaac), so run it in a throwaway env:

```bash
python -m venv .xacro_venv
.xacro_venv/Scripts/python -m pip install xacro
.xacro_venv/Scripts/python setup/expand_xacro.py
# -> assets/urdf/lbr_iiwa7_r800_description/iiwa7_r800.urdf
```

!!! note
    `xacro file > out.urdf` alone does **not** work here — it needs a ROS
    `ament_index` to resolve `$(find …)`. Use `expand_xacro.py`.

## 3. Import URDF → USD (needs Isaac Sim)

```powershell
python setup/import_iiwa.py
# -> assets/usd/iiwa7_r800.usd
```

This launches Isaac headless and runs the URDF importer. Expect a few benign GPU
warnings; success is the `assets/usd/iiwa7_r800.usd` file plus a
`configuration/` folder.

## What the importer does

- `fix_base=True` — the arm is bolted down (fixed base).
- `merge_fixed_joints=True` — folds the empty `lbr_link_ee` tool frame into
  `lbr_link_7` (which carries the collision geometry).
- `activate_contact_sensors=True` (set on the arm at load) — enables PhysX
  contact reporting so the flange force sensor works.

## Facts about this arm (used by the code)

- **Joints:** `lbr_A1` … `lbr_A7` (7 revolute).
- **End-effector body:** `lbr_link_7` — the last link with collision geometry.
- **Joint limits** (from `joint_limits.yaml`): ranges ±120°/±170°, speeds
  98–180 °/s. The controller clamps joint torque to the KUKA spec
  `[176, 176, 110, 110, 110, 40, 40]` N·m — note the **weak 40 N·m wrist**,
  which shapes where the arm can press (see [How it works](how-it-works.md)).
