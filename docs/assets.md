# Robot description

The iiwa 7 R800 URDF/meshes come from `lbr-stack/lbr_iiwa7_r800_description`
(chosen for parity with the real robot's ROS2 stack).

Fetch into `assets/urdf/` (run once, on any machine):

    git clone --depth 1 https://github.com/lbr-stack/lbr_iiwa7_r800_description \
        assets/urdf/lbr_iiwa7_r800_description

The description's top-level URDF/xacro is then imported to USD by
`setup/import_iiwa.py`. If the package ships xacro only, expand it first:

    xacro assets/urdf/lbr_iiwa7_r800_description/urdf/iiwa7_r800.urdf.xacro \
        > assets/urdf/iiwa7_r800.urdf

Confirm the exact file name inside the cloned repo; adjust `URDF_REL` in
`setup/import_iiwa.py` to match.
