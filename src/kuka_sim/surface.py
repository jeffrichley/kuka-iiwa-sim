"""A fixed, compliant contact surface (a soft 'table' the tool presses on)."""

def make_surface(prim_path="/World/Surface", pos=(0.60, 0.0, 0.70),
                 size=(0.05, 0.45, 0.55), stiffness=2000.0, damping=50.0):
    # A VERTICAL panel (thin in x); near face at x ~ 0.575, spanning z ~0.42-0.98.
    # The arm reaches forward and presses HORIZONTALLY (+x) into the face at its
    # natural forward-reach height (flange ~z=0.8), using the strong proximal
    # joints. Pressing a low table instead forces the arm to its reach limit where
    # the weak 40 N·m wrist joints saturate and it can't modulate a light force.
    # compliant_contact_stiffness makes the contact soft.
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
