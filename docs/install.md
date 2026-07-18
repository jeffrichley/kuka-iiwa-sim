# Install

One-time setup on a **native Windows 11** machine with an **RTX GPU** and a
recent NVIDIA driver.

!!! warning "Native Windows, not WSL2"
    Isaac Sim's RTX/Vulkan renderer does not initialize under WSL2, so run this
    natively on Windows. Python must be **3.11** exactly (Isaac Sim 5.1 / Isaac
    Lab 2.3 require it).

## 1. Run the installer

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File setup/install.ps1
```

This creates a Python 3.11 virtual environment (`env_isaaclab/`) and installs:

- **Isaac Sim 5.1** and **Isaac Lab 2.3** (from `pypi.nvidia.com`)
- a **CUDA-enabled** build of PyTorch (cu128) — the Windows default torch is
  CPU-only and fails GPU simulation with *"Torch not compiled with CUDA
  enabled"*, so the script replaces it
- this project, editable, plus its runtime deps

Activate the environment in each new shell:

```powershell
.\env_isaaclab\Scripts\Activate.ps1
```

## 2. Accept the Omniverse EULA

Isaac Sim requires a one-time acceptance of NVIDIA's Omniverse EULA. The scripts
set it non-interactively:

```
OMNI_KIT_ACCEPT_EULA=YES
PRIVACY_CONSENT=N          # declines telemetry
```

These are applied automatically inside `kuka_sim.sim_app.launch()` and
`setup/import_iiwa.py`, so you don't normally set them by hand. Review the
[NVIDIA Omniverse License](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
if you haven't.

## 3. Preflight

```powershell
python setup/check_env.py
```

Expected: `[OK] Python 3.11 and isaaclab import succeeded.`

The very first Isaac Sim launch compiles shaders (~30–60 s); later launches are
faster.

## Next

Build the robot asset → [Build the robot asset](build-asset.md).
