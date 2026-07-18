"""Environment preflight for the KUKA iiwa sim."""
import sys


def python_version_ok(version_info=None, required=(3, 11)):
    v = version_info if version_info is not None else sys.version_info
    return (v[0], v[1]) == required


def main():
    if not python_version_ok():
        print(f"[FAIL] Python {sys.version_info.major}.{sys.version_info.minor} "
              f"detected; Isaac Sim 5.1 / Isaac Lab 2.3 require exactly 3.11.")
        return 1
    try:
        import isaaclab  # noqa: F401
        print("[OK] Python 3.11 and isaaclab import succeeded.")
        return 0
    except ImportError:
        print("[FAIL] isaaclab not importable. Run setup/install.ps1 first.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
