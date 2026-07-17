"""Convert the iiwa 7 R800 URDF to USD via Isaac Lab's UrdfConverter.

Run on the Isaac machine after setup/install.ps1:
    python setup/import_iiwa.py
"""
import os

# Path relative to repo root. The plain URDF is produced by setup/expand_xacro.py
# and lives at the description package root so its relative `meshes/...` refs resolve.
URDF_REL = "assets/urdf/lbr_iiwa7_r800_description/iiwa7_r800.urdf"
USD_DIR_REL = "assets/usd"
USD_NAME = "iiwa7_r800.usd"


def resolve_paths(repo_root):
    """Pure path logic (no Isaac import) so it is unit-testable."""
    urdf = os.path.join(repo_root, URDF_REL)
    usd_dir = os.path.join(repo_root, USD_DIR_REL)
    return urdf, usd_dir, USD_NAME


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf, usd_dir, usd_name = resolve_paths(repo_root)
    if not os.path.exists(urdf):
        raise FileNotFoundError(
            f"URDF not found at {urdf}. Run setup/expand_xacro.py first "
            f"(see docs/assets.md to fetch the description).")
    os.makedirs(usd_dir, exist_ok=True)

    # Accept NVIDIA Omniverse EULA (declining telemetry) non-interactively.
    # setdefault so an explicit environment override always wins.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PRIVACY_CONSENT", "N")

    # Isaac imports must follow the app launch.
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    app = launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    cfg = UrdfConverterCfg(
        asset_path=urdf,
        usd_dir=usd_dir,
        usd_file_name=usd_name,
        fix_base=True,                 # iiwa is a fixed-base arm
        merge_fixed_joints=True,       # folds the empty lbr_link_ee tool frame away
        collider_type="convex_hull",
        self_collision=False,
        # gains.stiffness has no default (MISSING) and must be set to pass validation.
        # These position-drive gains are NOT load-bearing: at runtime IiwaArm's
        # ImplicitActuatorCfg(stiffness=0, damping=0) overrides them for torque control.
        # They only keep the standalone USD usable under position control.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=40.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[OK] wrote USD: {converter.usd_path}")
    app.close()


if __name__ == "__main__":
    main()
