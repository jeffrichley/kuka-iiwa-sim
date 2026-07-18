"""Boot Isaac Sim and return (app, SimulationContext)."""


def launch(headless=True, enable_cameras=True, dt=1.0 / 120.0, device="cuda:0"):
    import os

    # Accept NVIDIA Omniverse EULA (declining telemetry) non-interactively.
    # Must be set before any Isaac module is imported/instantiated, or the app
    # launch will hang on an interactive EULA prompt. setdefault so an explicit
    # environment override always wins.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PRIVACY_CONSENT", "N")

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=headless, enable_cameras=enable_cameras)
    simulation_app = app_launcher.app

    # Isaac modules must be imported AFTER the launcher starts.
    from isaaclab.sim import SimulationContext, SimulationCfg
    sim = SimulationContext(SimulationCfg(dt=dt, device=device))
    sim.set_camera_view([2.5, 2.5, 2.0], [0.4, 0.0, 0.3])
    return simulation_app, sim
