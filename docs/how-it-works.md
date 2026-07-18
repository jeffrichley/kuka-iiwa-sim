# How it works

The design, the control law, and the hard-won lessons about contact.

## Architecture

```
sim_app.launch()      boots Isaac headless, returns (app, SimulationContext)
IiwaArm               torque-controlled articulation + flange contact sensor
CartesianImpedance…   pose/force target -> clamped joint torques
CompliantSurface      a kinematic panel with tunable contact stiffness
SceneCamera           headless RGB capture
```

Demos and experiments consume only `IiwaArm` + the controller — never Isaac
internals. The controller depends on a small `ArmInterface`, so its math is
unit-tested against a `FakeArm` with no GPU.

## The control law

The arm is torque-controlled: the actuators are configured with
`stiffness = damping = 0`, so commanded joint efforts pass straight through. On
top of that, the Cartesian impedance controller computes

$$\tau = J^\top\,(K\,x_e - D\,\dot{x}) + g(q)$$

- \(x_e\) — pose error (position + axis-angle orientation), world frame
- \(J\) — end-effector Jacobian; \(g(q)\) — gravity compensation
- \(K, D\) — per-axis task-space stiffness / damping
- force-selected axes replace their impedance term with a feed-forward wrench

Torque is clamped to the iiwa spec. Gravity compensation is added **before**
clamping, so the clamp bounds *total* commanded torque. With perfect gravity
comp the arm floats, so in free space it converges to the pose target.

## The contact story (the interesting part)

Getting a *gentle, steady* contact force was the hard part, and it taught us the
arm's real envelope.

### Pressing down doesn't work — and that's physical

The iiwa 7 R800's **wrist joints are only 40 N·m** (vs 176 N·m at the shoulder).
When the arm reaches *down and out* to a low table, it folds so those weak wrist
joints bear the press — they saturate almost instantly. The arm can only **slam**
(45–90 N, torque-saturated) or **float** off; there is no clean 10 N regime. This
is true of the real robot too — it's the arm's workspace boundary, not a bug.

**Fix:** press in the **dexterous zone** — a vertical panel in front of the arm,
pressed *horizontally* with the strong proximal joints. There the arm tracks a
forward pose to ~1 mm and can modulate a light force.

### Feed-forward force control floats off

Commanding the contact axis with a small feed-forward force (e.g. +10 N) doesn't
hold contact: the force is too weak to overcome the arm's pose-restoring dynamics
and the flange backs off. A stronger position press holds fine, but isn't a
force setpoint.

**Fix:** **impedance-based force control.** Command a fixed press *depth* into the
**compliant** workpiece; the contact converts depth into a steady force. It's
stable, and the depth→force map is monotonic and calibratable:

| press depth x | steady force |
|---|---|
| 0.42 | ~30 N |
| 0.44 | ~40 N |
| 0.46 | ~53 N |

### Press with a tool tip, not the wrist

Without a tool, the *side of the wrist* contacts the surface — not a defined
point. A **probe** is attached to the flange (`scene.py::_add_flange_probe`,
spawned as a child of `lbr_link_7` so its contacts register on the flange
sensor), and the flange is **aimed forward** (`+90°` about Y, so the flange +Z /
tool axis points at the block) so the **tool tip** does the pressing — matching a
real end-effector (probe / scalpel).

### Penetration vs. force are coupled

With a compliant block, penetration ≈ force / contact-stiffness — a *soft* block
forces the tip to sink in deeply to build force (looks like it's going through).
A *stiff* block (stiffness 8000 + matching damping) keeps the indent to a few mm
so the tip presses **on** the face, and the force is rock-steady. Soften the
block only if you want the tool to press **into** a malleable surface.

### A saturated axis drives slow instability

Even in contact, the press first drifted upward (10 → 24 N) over several seconds.
Cause: the pose target's **z was unreachable** (0.5 while the flange rests at
0.8), so the z-axis permanently commanded *saturated* downward torque, which fed
a slow rocking against the compliant panel.

**Fix:** set the pose target's z to the flange's **natural reach height** (~0.8)
so the z-axis de-saturates. The press then converges to a steady value
(verified: 10.25 N mean, flat).

## Takeaways for your experiments

- Press **forward, mid-height**, not down at reach.
- Keep pose targets **reachable** (especially z) so no axis saturates.
- Set force via **press depth** into the compliant surface; re-sweep if you
  change surface stiffness.
- The `CompliantSurface` is a swap point — a future `DeformableSurface` (FEM soft
  body) can replace it for visible deformation. Cutting / material removal is not
  modeled (PhysX doesn't do topology change out of the box).
