"""A fixed, compliant contact surface (a soft 'table' the tool presses on)."""

def make_surface(prim_path="/World/Surface", pos=(0.72, 0.0, 0.62),
                 size=(0.24, 0.40, 0.40), stiffness=8000.0, damping=120.0):
    # A solid workpiece BLOCK in front of the arm; near face at x ~ 0.60,
    # mid-height. The arm reaches forward with a flange PROBE (see scene.py) and
    # presses the probe tip HORIZONTALLY (+x) into the face, using the strong
    # proximal joints. Pressing a low table instead forces the arm to its reach
    # limit where the weak 40 N·m wrist joints saturate. A softer
    # compliant_contact_stiffness gives a gentler, steadier press force.
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    cfg = RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8,
                compliant_contact_stiffness=stiffness,   # >0 enables compliant contact
                compliant_contact_damping=damping),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.55, 0.55)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos)),
    )
    return RigidObject(cfg)
