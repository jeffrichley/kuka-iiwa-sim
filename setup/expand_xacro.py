"""Expand the iiwa 7 R800 xacro into a plain URDF Isaac can import.

The description ships as xacro (``lbr_iiwa7_r800.urdf.xacro`` + a macro that
loads ``config/joint_limits.yaml``). Isaac's URDF importer needs plain URDF, so
this expands it. It deliberately avoids a ROS install: ``$(find <pkg>)`` refs are
substituted with the package's absolute path in temporary copies of the xacro
files, so xacro never needs ``ament_index``/``rospkg`` to resolve them. Mesh refs
(``package://<pkg>/meshes/...``) are rewritten to paths relative to the emitted
URDF, which is written at the package root so ``meshes/...`` resolves beside it.

Requires the ``xacro`` package (``pip install xacro``) — NOT Isaac. Run it in any
Python env (e.g. the throwaway ``.xacro_venv``); it does not import Isaac Sim.

Usage:
    python setup/expand_xacro.py
    python setup/expand_xacro.py --pkg-dir <path> --out <path>
"""
import argparse
import os

PKG_NAME = "lbr_iiwa7_r800_description"
DEFAULT_PKG_DIR = f"assets/urdf/{PKG_NAME}"
TOP_XACRO = "urdf/lbr_iiwa7_r800.urdf.xacro"
MACRO_XACRO = "urdf/lbr_iiwa7_r800_macro.xacro"
OUT_NAME = "iiwa7_r800.urdf"


def expand(pkg_dir, out_path):
    """Expand the package's top-level xacro to a plain URDF at out_path."""
    import xacro

    pkg_dir = os.path.abspath(pkg_dir).replace(os.sep, "/")
    find = f"$(find {PKG_NAME})"
    urdfd = os.path.join(pkg_dir, "urdf")
    macro_tmp = os.path.join(urdfd, "_macro_sub.xacro")
    top_tmp = os.path.join(urdfd, "_top_sub.xacro")

    # Substitute $(find pkg) -> absolute path in temp copies (originals stay pristine),
    # and repoint the include at the substituted macro.
    with open(os.path.join(pkg_dir, MACRO_XACRO), encoding="utf-8") as f:
        macro = f.read().replace(find, pkg_dir)
    with open(macro_tmp, "w", encoding="utf-8") as f:
        f.write(macro)
    with open(os.path.join(pkg_dir, TOP_XACRO), encoding="utf-8") as f:
        top = f.read().replace(find, pkg_dir).replace(
            "lbr_iiwa7_r800_macro.xacro", "_macro_sub.xacro")
    with open(top_tmp, "w", encoding="utf-8") as f:
        f.write(top)

    try:
        urdf = xacro.process_file(top_tmp).toprettyxml(indent="  ")
    finally:
        for f in (macro_tmp, top_tmp):
            if os.path.exists(f):
                os.remove(f)

    # package://<pkg>/meshes/... -> meshes/... (relative to the URDF at the pkg root)
    urdf = urdf.replace(f"package://{PKG_NAME}/", "")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(urdf)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg-dir", default=DEFAULT_PKG_DIR,
                        help="Path to the cloned description package.")
    parser.add_argument("--out", default=None,
                        help="Output URDF path (default: <pkg-dir>/iiwa7_r800.urdf).")
    args = parser.parse_args()

    top = os.path.join(args.pkg_dir, TOP_XACRO)
    if not os.path.exists(top):
        raise FileNotFoundError(
            f"xacro not found at {top}. Clone the description first (see docs/assets.md).")
    out = args.out or os.path.join(args.pkg_dir, OUT_NAME)
    written = expand(args.pkg_dir, out)
    print(f"[OK] wrote plain URDF: {written}")


if __name__ == "__main__":
    main()
