# Robot description

The iiwa 7 R800 URDF/meshes come from `lbr-stack/lbr_iiwa7_r800_description`
(chosen for parity with the real robot's ROS2 stack). It is **fetched**, not
committed — `assets/urdf/` is gitignored.

## 1. Fetch (run once, any machine)

    git clone --depth 1 https://github.com/lbr-stack/lbr_iiwa7_r800_description \
        assets/urdf/lbr_iiwa7_r800_description

## 2. Expand xacro → plain URDF

The description ships as **xacro** (`urdf/lbr_iiwa7_r800.urdf.xacro` + a macro
that loads `config/joint_limits.yaml`), which Isaac's importer cannot read
directly. `setup/expand_xacro.py` expands it to a plain URDF **without needing a
ROS install** — it substitutes `$(find <pkg>)` refs with the package's absolute
path and rewrites `package://` mesh refs to paths relative to the emitted URDF.

It needs the `xacro` pip package (NOT Isaac), so run it in a throwaway env:

    python -m venv .xacro_venv
    .xacro_venv/Scripts/python -m pip install xacro
    .xacro_venv/Scripts/python setup/expand_xacro.py
    # -> assets/urdf/lbr_iiwa7_r800_description/iiwa7_r800.urdf

(The naive `xacro file > out.urdf` does NOT work here — it requires
`ament_index`/a ROS install to resolve `$(find …)`. Use `expand_xacro.py`.)

## 3. Import URDF → USD (needs Isaac Sim)

    python setup/import_iiwa.py
    # -> assets/usd/iiwa7_r800.usd

## Facts about this description (used by the sim code)

- **Joints:** `lbr_A1` … `lbr_A7` (7 revolute) + `lbr_joint_ee` (fixed).
- **Links:** `lbr_link_0` … `lbr_link_7`, plus an empty tool frame `lbr_link_ee`.
- **End-effector body for control/contact:** `lbr_link_7` — the last link *with
  collision geometry* (so the contact sensor registers surface presses).
  `lbr_link_ee` is an empty offset frame with no collision; `merge_fixed_joints`
  folds it away during import.
- **No** gazebo / ros2_control / transmission tags — a clean import.
