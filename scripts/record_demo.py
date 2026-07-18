"""Deliverable E: record the force demo — MP4 + force-vs-time plot.

Runs the same force-regulation core as force_demo.py with the camera rolling,
then encodes the frames to out/iiwa_force_demo.mp4 and saves the logged contact
force to out/force_trace.png.
"""
import os
import sys

# scripts/ is sys.path[0] when run directly, so import the sibling module by name.
# kuka_sim resolves via the editable install.
from force_demo import run, NOMINAL_FORCE_N, USD
from kuka_sim.sim_app import launch
from kuka_sim.scene import build_scene
from kuka_sim.camera import SceneCamera
from kuka_sim.recording import frames_to_mp4

OUT_DIR = "out"
CAPTURE_EVERY = 4          # 120 Hz sim / 4 -> 30 fps video


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    app, sim = launch(headless=True, enable_cameras=True)
    handles = build_scene(sim, USD)
    cam = SceneCamera(pos=(1.9, 1.9, 1.2), target=(0.42, 0.0, 0.62))
    sim.reset(); handles["arm"].initialize(); cam.initialize()

    frames = []

    def on_step(i):
        cam.update()
        if i % CAPTURE_EVERY == 0:
            frames.append(cam.capture())

    log = run(sim=sim, app=app, handles=handles, steps=2500, on_step=on_step)

    mp4 = os.path.join(OUT_DIR, "iiwa_force_demo.mp4")
    png = os.path.join(OUT_DIR, "force_trace.png")
    frames_to_mp4(frames, mp4, fps=30)
    log.save_plot(png, target=NOMINAL_FORCE_N)
    print(f"[OK] wrote {mp4} ({len(frames)} frames) and {png}", flush=True)
    app.close()
    os._exit(0)   # Isaac leaves non-daemon threads; force a clean exit


if __name__ == "__main__":
    main()
