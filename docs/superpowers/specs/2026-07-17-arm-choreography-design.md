# Design: Arm Choreography ("dance to music") — bonus module

**Date:** 2026-07-17
**Status:** Design captured; queued as a BONUS module, built after the core
force-control tasks (9–15) land.
**Relationship to core:** additive and isolated — lives under
`src/kuka_sim/music/` + `scripts/dance_demo.py`, reuses `IiwaArm`,
`SceneCamera`, and `frames_to_mp4`, and touches NONE of the digital-twin
force-control path.

## Goal

A **general** pipeline: take ANY audio file, analyze it, choreograph the iiwa
7 R800 to it, and render a music video — the arm moving to the beat, a camera
track synced to the music, and the song muxed onto the final MP4.

Showcase inputs: **"The Blue Danube"** (waltz, 3/4 sway) and **"Also sprach
Zarathustra"** (epic build). These are just example inputs — the pipeline is
song-agnostic.

## Non-goals
- Not real-time / interactive — the whole trajectory is precomputed from the
  audio timeline, then played back deterministically at sim dt.
- Not breakdancing — the iiwa has real joint velocity/torque limits, so the
  choreography is graceful (waltz/tai-chi/conductor), not snappy. Moves are
  authored in JOINT space within velocity limits to stay smooth and dodge
  Cartesian singularities.

## Architecture

```
  src/kuka_sim/music/
    audio_features.py   librosa analysis -> per-sim-tick feature timeline
    choreographer.py    features -> (arm joint trajectory, camera track)
    moves.py            a small library of named joint-space poses/gestures
  scripts/
    dance_demo.py       CLI: --song <path> --camera {hero,cinematic,both} --out
  assets/audio/         input songs (gitignored)
```

### audio_features.py (song-agnostic)
Uses `librosa`. Extracts and resamples to the sim dt (1/120 s) so sim-frame N
== song-time N·dt:
- beat times + tempo (`librosa.beat.beat_track`)
- RMS energy envelope (motion/camera intensity)
- band energies: bass / mid / treble (mel spectrogram split)
- onsets (accent triggers)
Returns a `FeatureTrack` sampled at sim dt.

### choreographer.py
Maps the FeatureTrack to two precomputed tracks:
- **Arm trajectory** (joint-space, N×7): a baseline sway driven by RMS +
  band→joint mapping (bass→base sway, treble→wrist), with `moves.py` poses
  triggered on beats and interpolated smoothly (the impedance controller — or
  direct joint targets — blends between them). Clamped to iiwa velocity limits.
- **Camera track** (N poses): `hero` = a single fixed hero pose; `cinematic`
  = slow orbit baseline + RMS-driven push-in on musical peaks. Restrained so it
  never fights the arm.

### dance_demo.py
1. `audio_features(song)` -> FeatureTrack
2. `choreograph(track, camera_mode)` -> (arm_traj, camera_tracks)
3. run the sim: for each tick, apply arm target, set camera pose, capture frame
   (one frame stream per camera mode)
4. `frames_to_mp4` per camera -> silent MP4(s)
5. **mux audio**: `ffmpeg -i clip.mp4 -i <song> -c:v copy -c:a aac -shortest
   <out>.mp4` (ffmpeg ships with imageio-ffmpeg)
`--camera both` renders a `_hero.mp4` and a `_cinematic.mp4` per song.

## Dependencies
- `librosa` (+ its deps) — add to an optional `[music]` extra in pyproject so
  the core install stays lean.
- ffmpeg — already present via `imageio-ffmpeg`.

## Deliverable
For each showcase song: `out/<song>_hero.mp4` and `out/<song>_cinematic.mp4`,
each the arm dancing with the song playing. Plus: drop any file in
`assets/audio/` and `python scripts/dance_demo.py --song assets/audio/<file>`
produces the same.

## Open questions (defer to build time)
- Move library richness: start with ~5–8 poses; expand if it reads thin.
- Whether to drive the arm via the Cartesian impedance controller (smooth,
  compliant blend) or direct joint-position targets (simplest, safest). Likely
  joint targets for the dance; the controller is overkill unless we want
  compliance to a prop.
