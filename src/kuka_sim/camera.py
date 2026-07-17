"""A fixed scene camera that captures RGB frames headless."""
import numpy as np


class SceneCamera:
    def __init__(self, prim_path="/World/DemoCam", pos=(2.2, 1.6, 1.4),
                 target=(0.55, 0.0, 0.3), width=960, height=540, dt=1.0 / 120.0,
                 device="cuda:0"):
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        self.dt = dt
        self.device = device
        cfg = CameraCfg(
            prim_path=prim_path,
            update_period=0.0, height=height, width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                             clipping_range=(0.05, 20.0)),
        )
        self.camera = Camera(cfg)
        self._pos = np.array(pos, dtype=float)
        self._target = np.array(target, dtype=float)

    def initialize(self):
        # aim the camera after the physics view exists
        import torch
        # eyes/targets must be on the camera's device — set_world_poses_from_view
        # runs a cross-product on cuda, so CPU tensors trigger a device mismatch.
        self.camera.set_world_poses_from_view(
            torch.tensor([self._pos], dtype=torch.float32, device=self.device),
            torch.tensor([self._target], dtype=torch.float32, device=self.device))

    def update(self):
        self.camera.update(self.dt)

    def capture(self):
        rgb = self.camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:    # drop alpha if RGBA
            rgb = rgb[..., :3]
        return rgb
