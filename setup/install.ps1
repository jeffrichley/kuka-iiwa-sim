# setup/install.ps1 — one-time environment setup for the KUKA iiwa sim (Windows, native).
# Requires: Python 3.11 on PATH, an NGC account for Isaac assets, a recent NVIDIA driver.
$ErrorActionPreference = "Stop"

Write-Host "== Creating Python 3.11 venv (env_isaaclab) =="
py -3.11 -m venv env_isaaclab
.\env_isaaclab\Scripts\Activate.ps1

python -c "import sys; assert sys.version_info[:2]==(3,11), 'need Python 3.11'"

Write-Host "== Upgrading pip and installing Isaac Sim 5.1 + Isaac Lab 2.3 =="
python -m pip install --upgrade pip
# Isaac Sim 5.1 (pip) — pulls the RTX runtime; ~large download.
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# Isaac Lab 2.3 (stable) as the robotics API layer.
pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

Write-Host "== Installing project (editable) + dev deps =="
pip install -e ".[dev]"

Write-Host "== Preflight =="
python setup/check_env.py

Write-Host "== Done. Next: python setup/import_iiwa.py to build the arm USD. =="
