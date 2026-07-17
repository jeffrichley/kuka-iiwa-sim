"""Torque-controlled iiwa 7 R800 wrapper implementing ArmInterface."""
import numpy as np


class IiwaArm:
    """Wraps an Isaac Lab Articulation. Direct-torque control (implicit actuator
    with stiffness=damping=0) + a ContactSensor on the flange for contact force.
    """

    N_JOINTS = 7

    def __init__(self, usd_path, prim_path="/World/Robot", ee_body="lbr_link_7",
                 surface_prim="/World/Surface", device="cuda:0", dt=1.0 / 120.0):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        self.device = device
        self.dt = dt
        self.ee_body = ee_body
        self._ee_idx = None

        robot_cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path, activate_contact_sensors=True),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0, damping=0.0,      # pass-through: raw torque control
                    effort_limit=320.0, velocity_limit=100.0),
            },
        )
        self.robot = Articulation(robot_cfg)

        contact_cfg = ContactSensorCfg(
            prim_path=f"{prim_path}/{ee_body}",
            update_period=0.0, history_length=4, track_pose=True,   # history → smoothing
            filter_prim_paths_expr=[surface_prim],
        )
        self.contact = ContactSensor(contact_cfg)

    # --- lifecycle ---
    def initialize(self):
        """Call once after sim.reset() so the physics view exists."""
        self._ee_idx = self.robot.find_bodies(self.ee_body)[0][0]

    def write(self):
        self.robot.write_data_to_sim()

    def update(self):
        self.robot.update(self.dt)
        self.contact.update(self.dt)

    def reset(self):
        self.robot.reset()
        self.contact.reset()   # IiwaArm owns both; clear stale contact-force history too

    # --- ArmInterface ---
    def get_joint_velocities(self):
        return self.robot.data.joint_vel[0].detach().cpu().numpy()

    def get_ee_pose(self):
        pos = self.robot.data.body_pos_w[0, self._ee_idx].detach().cpu().numpy()
        quat = self.robot.data.body_quat_w[0, self._ee_idx].detach().cpu().numpy()
        return pos, quat

    def get_jacobian(self):
        # World-frame Jacobian. Fixed base at world origin (identity root orientation)
        # → world frame == base frame, and pose error is also computed in world frame
        # (body_pos_w / body_quat_w), so the whole control law is frame-consistent.
        # (NVIDIA's OSC tutorial rotates J into the base frame; unnecessary here.)
        jac_body_idx = self._ee_idx - 1        # fixed base: root excluded from jacobians
        J = self.robot.root_physx_view.get_jacobians()[0, jac_body_idx, :, :self.N_JOINTS]
        return J.detach().cpu().numpy()

    def get_gravity_torque(self):
        g = self.robot.root_physx_view.get_gravity_compensation_forces()[0, :self.N_JOINTS]
        return g.detach().cpu().numpy()

    def set_joint_efforts(self, tau):
        import torch
        tau = np.asarray(tau, float)
        if tau.shape != (self.N_JOINTS,):
            raise ValueError(f"tau must be shape (7,), got {tau.shape}")
        t = torch.as_tensor(tau, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.robot.set_joint_effort_target(t)

    # --- extras (demos) ---
    def get_joint_positions(self):
        return self.robot.data.joint_pos[0].detach().cpu().numpy()

    def get_ee_force(self):
        """Contact force (N, world frame) on the flange, smoothed over the sensor
        history. Mirrors NVIDIA's OSC tutorial: mean over history, max over bodies."""
        import torch
        hist = self.contact.data.net_forces_w_history      # (1, T, B, 3)
        mean_over_time = torch.mean(hist, dim=1)            # (1, B, 3)
        f, _ = torch.max(mean_over_time, dim=1)            # (1, 3) strongest-contact body
        return f[0].detach().cpu().numpy()
