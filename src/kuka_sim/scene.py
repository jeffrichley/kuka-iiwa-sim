"""Assemble the demo scene: ground, light, arm (+ flange probe), workpiece."""
from kuka_sim.robot import IiwaArm
from kuka_sim.surface import make_surface

SURFACE_PRIM = "/World/Surface"
ROBOT_PRIM = "/World/Robot"
EE_BODY = "lbr_link_7"


def _add_flange_probe(prim_path, length=0.16, radius=0.012, mount=0.10):
    """Attach a rigid probe/tool to the flange (link_7) so the arm presses with a
    defined tool TIP, not the side of the wrist. Spawned as a child of the flange
    rigid body, so its collisions are part of link_7 and register on the flange
    contact sensor, and it extends along the flange +Z (tool/approach axis)."""
    import isaaclab.sim as sim_utils
    probe = sim_utils.CylinderCfg(
        radius=radius, height=length, axis="Z",
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
    )
    probe.func(prim_path, probe, translation=(0.0, 0.0, mount))


def build_scene(sim, usd_path, surface_kwargs=None, with_probe=True, with_surface=True):
    """Ground + light + arm, plus (optionally) the force-demo workpiece and flange
    probe. Dance scenes pass with_probe=False, with_surface=False for a bare arm."""
    import isaaclab.sim as sim_utils

    # ground + dome light — documented standalone spawn idiom: cfg.func(path, cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    surface = make_surface(prim_path=SURFACE_PRIM, **(surface_kwargs or {})) if with_surface else None
    arm = IiwaArm(usd_path=usd_path, prim_path=ROBOT_PRIM, ee_body=EE_BODY,
                  surface_prim=(SURFACE_PRIM if with_surface else None))
    if with_probe:
        _add_flange_probe(f"{ROBOT_PRIM}/{EE_BODY}/probe")
    return {"arm": arm, "surface": surface}
