"""Assemble the demo scene: ground, light, arm, compliant surface."""
from kuka_sim.robot import IiwaArm
from kuka_sim.surface import make_surface

SURFACE_PRIM = "/World/Surface"


def build_scene(sim, usd_path, surface_kwargs=None):
    import isaaclab.sim as sim_utils

    # ground + dome light — documented standalone spawn idiom: cfg.func(path, cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    surface = make_surface(prim_path=SURFACE_PRIM, **(surface_kwargs or {}))
    arm = IiwaArm(usd_path=usd_path, surface_prim=SURFACE_PRIM)
    return {"arm": arm, "surface": surface}
