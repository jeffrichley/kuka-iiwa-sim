"""A fixed, compliant contact surface (a soft 'table' the tool presses on)."""

def make_surface(prim_path="/World/Surface", pos=(0.55, 0.0, 0.25),
                 size=(0.4, 0.4, 0.04), stiffness=2000.0, damping=50.0):
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
