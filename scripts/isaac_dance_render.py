"""Isaac beauty render: two KUKAs play the baked mimic trajectory on a dark,
spotlit stage, recorded with the source clip inset.

Loads out/<traj>.npz (qr, ql, sim_fps from gen_mimic_traj.py), spawns two
position-controlled iiwa arms mounted shoulder-width apart, sets a stage look
(dark glossy floor + dim ambient + a warm spotlight), plays the trajectory and
films it.

Usage (Isaac venv):
    env_isaaclab/Scripts/python.exe scripts/isaac_dance_render.py \
        --traj out/yt1_traj.npz --clip out/yt1_seg.mp4 --seconds 8 \
        --out out/isaac_dance.mp4
"""
import argparse
import os
import numpy as np

USD = "assets/usd/iiwa7_r800.usd"
ARM_OFFSET = 0.55          # half the distance between the two bases (metres)
CAPTURE_EVERY = 2          # 60 Hz sim / 2 -> 30 fps


def _build_arm(prim_path, pos):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, joint_pos={".*": 0.0}),
        actuators={"arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=600.0, damping=50.0,
            effort_limit=320.0, velocity_limit=100.0)},
    )
    return Articulation(cfg)


def _build_stage():
    import isaaclab.sim as sim_utils
    # dark glossy floor (reads as a reflective stage)
    floor = sim_utils.CuboidCfg(
        size=(30.0, 30.0, 0.1),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.02, 0.03), roughness=0.25, metallic=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    floor.func("/World/floor", floor, translation=(0.0, 0.0, -0.05))
    # dim cool ambient so the surround stays dark
    dome = sim_utils.DomeLightCfg(intensity=90.0, color=(0.15, 0.17, 0.25))
    dome.func("/World/dome", dome)
    # warm key "spotlight" tight over the arms
    spot = sim_utils.SphereLightCfg(intensity=350000.0, radius=0.22,
                                    color=(1.0, 0.93, 0.8))
    spot.func("/World/spot", spot, translation=(0.6, 0.0, 2.4))
    # front fill on the camera side (-x now)
    fill = sim_utils.SphereLightCfg(intensity=120000.0, radius=0.4, color=(0.9, 0.92, 1.0))
    fill.func("/World/fill", fill, translation=(-2.0, -0.7, 1.5))
    # blue rim from behind (opposite the camera, +x) for separation
    rim = sim_utils.SphereLightCfg(intensity=45000.0, radius=0.3, color=(0.35, 0.5, 1.0))
    rim.func("/World/rim", rim, translation=(1.8, 0.0, 1.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--clip", default=None, help="source clip for PiP + audio")
    ap.add_argument("--seconds", type=float, default=None, help="cap render length")
    ap.add_argument("--out", default="out/isaac_dance.mp4")
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    data = np.load(args.traj)
    qr, ql = data["qr"], data["ql"]
    sim_fps = float(data["sim_fps"])
    n = len(qr)
    if args.seconds is not None:
        n = min(n, int(args.seconds * sim_fps))
    print(f"[render] {n} sim frames @ {sim_fps:.0f}fps", flush=True)

    from kuka_sim.sim_app import launch
    from kuka_sim.camera import SceneCamera
    from kuka_sim.dance.recorder import record
    import torch

    app, sim = launch(headless=True, enable_cameras=True, dt=1.0 / sim_fps)
    _build_stage()
    arm_r = _build_arm("/World/RobotR", (0.0, -ARM_OFFSET, 0.0))
    arm_l = _build_arm("/World/RobotL", (0.0, ARM_OFFSET, 0.0))
    # camera on the -x side, matching the approved preview's viewpoint (azim~195)
    # so toward-camera reaches read correctly instead of pointing away.
    cam = SceneCamera(pos=(-3.7, -1.0, 1.35), target=(0.0, 0.0, 0.55),
                      dt=1.0 / sim_fps)
    sim.reset()
    cam.initialize()

    dev = arm_r.device
    frames = []
    for i in range(n):
        arm_r.set_joint_position_target(torch.tensor(qr[i], dtype=torch.float32, device=dev).unsqueeze(0))
        arm_l.set_joint_position_target(torch.tensor(ql[i], dtype=torch.float32, device=dev).unsqueeze(0))
        arm_r.write_data_to_sim(); arm_l.write_data_to_sim()
        sim.step()
        arm_r.update(1.0 / sim_fps); arm_l.update(1.0 / sim_fps)
        if i % CAPTURE_EVERY == 0:
            cam.update()
            frames.append(cam.capture())

    print(f"[render] captured {len(frames)} frames; encoding", flush=True)
    record(frames, args.out, fps=int(round(sim_fps / CAPTURE_EVERY)),
           audio_path=args.clip, pip_video_path=args.clip)
    print(f"[OK] wrote {args.out}", flush=True)

    import sys, threading
    sys.stdout.flush(); sys.stderr.flush()
    t = threading.Thread(target=app.close, daemon=True); t.start(); t.join(timeout=15)
    os._exit(0)


if __name__ == "__main__":
    main()
