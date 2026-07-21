# The dance module

A bonus on top of the force-control twin: make the arm **dance**. It grew into a
full motion pipeline that retargets a human dancer onto the KUKA, joint for
joint, and renders two arms performing a real routine.

## Evolution of Dance — two KUKAs

Two iiwa arms perform the entire six-minute *Evolution of Dance*, every move
retargeted from the original dancer onto the arms in 3D:

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: 8px;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
    src="https://www.youtube.com/embed/NMuR0vQ6Eag"
    title="Two KUKA iiwa arms dance the Evolution of Dance"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen></iframe>
</div>

10,805 simulation frames, start to finish — the whole routine, no cuts, with the
source clip inset for comparison.

## Three ways to drive the arm

| Mode | Input | Script |
|---|---|---|
| **Backend** | a canned joint trajectory | `dance_backend_demo.py` |
| **Music (V1)** | a song → beat-synced choreography | `dance_music_demo.py` |
| **Video mimic (V2)** | a dance video → the arm mimics the dancer | the mimic pipeline below |

The backend (`kuka_sim.dance`) is the shared spine: `trajectory` builds and
joint-limits a motion, `player` streams targets into the sim, `recorder` films it
and muxes audio + an optional picture-in-picture clip.

## The mimic pipeline (V2)

This is the flagship. A human dancer is mapped onto the **entire arm** — torso,
shoulder, elbow, wrist, hand — in full 3D, so the robot follows her limbs in
depth, not just side to side.

```
video ──▶ extract_pose ──▶ gen_mimic_traj ──▶ isaac_dance_render ──▶ mp4
         (.pose_venv)      (.preview_venv)     (env_isaaclab)
          MediaPipe          retarget → IK        two KUKAs
```

1. **Pose** — MediaPipe extracts 33 3D body landmarks per frame. This runs in
   an **isolated `.pose_venv`** (MediaPipe pulls opencv-contrib, which would
   clobber Isaac's `cv2`) and hands off a plain `.npz`.
2. **Retarget + bake** — for each frame, a bounded least-squares solve finds the
   7 joint angles that best align the arm's real link directions (from URDF
   forward kinematics) with the dancer's limb directions. Both a **right** and a
   **left** arm are solved, smoothed, and resampled to the sim rate. Runs in
   `.preview_venv` (ikpy + scipy).
3. **Render** — two position-controlled KUKAs on a spotlit stage play the baked
   trajectory; the source clip is inset and its audio muxed in.

```powershell
# 1) pose  (isolated pose venv)
.pose_venv/Scripts/python.exe scripts/extract_pose.py `
    --video assets/video/dance.mp4 --out out/dance_poses.npz

# 2) bake the two-arm joint trajectory  (preview venv)
.preview_venv/Scripts/python.exe scripts/gen_mimic_traj.py `
    --poses out/dance_poses.npz --start 20 --seconds 40 `
    --sim-fps 60 --out out/dance_traj.npz

# 3) beauty render  (Isaac venv)
env_isaaclab/Scripts/python.exe scripts/isaac_dance_render.py `
    --traj out/dance_traj.npz --clip out/dance_clip.mp4 `
    --out out/isaac_dance.mp4
```

!!! warning "Three isolated venvs — on purpose"
    MediaPipe and the preview stack both pin `numpy < 2`, but Isaac is compiled
    against the **numpy 1.26 ABI** and crashes on numpy 2. Keeping pose,
    preview, and Isaac in separate venvs — handing plain `.npz` files between
    them — is what keeps all three from clobbering each other.

## Iterate without Isaac

A full Isaac beauty render is slow. `preview_trajectory.py` renders the arm as a
dots-and-lines stick figure in **matplotlib 3D** in seconds, so choreography,
amplitude, camera, and the mimic mapping can be tuned fast — then the final
beauty pass runs in Isaac once.

```powershell
.preview_venv/Scripts/python.exe scripts/preview_trajectory.py `
    --mode mimic --poses out/dance_poses.npz --side both `
    --seconds 20 --out out/preview_mimic.mp4
```

The stick figure draws the robot's **real** joint positions (IK from the URDF),
color-coded so each robot link matches the human segment it tracks. It
approximates the motion, not the physics (Isaac adds impedance lag).

## How well does it track?

`joint_distance_analysis.py` measures, per frame, how far each robot joint lands
from the dancer's — anchored at the shoulder and normalized by upper-arm length —
and writes a distance-over-time plot plus a 3D overlay. Moving from a
side-to-side mapping to the full-3D solve cut mean tracking error from ~1.5 to
~0.5 upper-arm lengths.

## Where it lives

```
src/kuka_sim/dance/
  trajectory.py  player.py  recorder.py     backend spine
  music/         features, moves, choreographer   (V1)
  video/         pose, retarget, mimic, saliency   (V2)
scripts/
  dance_backend_demo.py  dance_music_demo.py       demos
  extract_pose.py  gen_mimic_traj.py  isaac_dance_render.py   mimic pipeline
  preview_trajectory.py  joint_distance_analysis.py           iterate + measure
```
