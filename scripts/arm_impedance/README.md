# Wrist-less (variable) Cartesian impedance on key insertion

What the arm's own impedance can and cannot do for a rigidly held key, with the controller law and gains HSA-UMI
runs on the robot. The question behind it: while the wrist is down, which of its jobs can the arm's controller take
over, and which gains should change.

- [Setup](#setup)
- [Studies](#studies)
- [Results](#results)
- [What to change on the robot](#what-to-change-on-the-robot)
- [Reproducing everything](#reproducing-everything)

## Setup

| part | what |
|---|---|
| robot | `FR3StewartAloha` (`--robot fer` for the Panda), the `KeyInsertion` task: 10 g key, 0.25 mm clearance per side, 2 mm chamfer, spring plunger 4 → 10 N over the last 4 mm |
| "wrist-less" | the Stewart wrist held fully contracted (`wrist_z = −1`): 100 kN/m legs, 100 kN/m lateral and 417 N·m/rad in tilt, 50× the arm's spring, so the arm's impedance is the only compliance that matters |
| arm | the robot's law (`arm_controller.py`: K 2000 N/m / 150 N·m/rad, D 89 / 7, clip 10 mm / 0.05 rad → 20 N ceiling, damped-projector nullspace, 1 N·m/ms slew, 6-waypoint chunk reference), FR3 gains; robosuite's OSC (kp 150, inertia-shaped, no ceiling) as the comparison the sim teleop used until now |
| expert | the scripted oracle (`key_insertion/oracle.py`): holds the wrist, moves above where it believes the keyhole is, lowers at 10 mm/s, stops when seated or 15 mm past the seat ("stuck"). The HSA law has no force stop: its clip is the stop. OSC keeps the oracle's 40 N stop |
| errors | the oracle's belief is offset by a perception error: dx across the slot, dyaw about the keyhole axis. Those are the two the stiff wrist fails on in the key-insertion study |
| live gains | the sweeps and schedules change `kp`, `damping` and the clips on the live controller between and during episodes, as `ros2 param set` does on the robot (`apply_gains`). robosuite rebuilds the controllers on every reset, so this happens after the reset |

## Studies

| study | sweep | episodes |
|---|---|---|
| grid | OSC vs the HSA law over dx ∈ [−2, 2] mm × dyaw ∈ [−6, 6]°, 9 × 9 | 162 |
| rot_k | rotational stiffness 1.5, 5, 15, 50, 150, 450 N·m/rad (damping ratio kept at the robot's 0.29), over the 25-case error set (dx ∈ ±1.5 mm, dyaw ∈ ±4°) | 150 |
| trans_k | translational stiffness 500 … 8000 N/m with the ceiling kept at 20 N (clip = 20 N / K) | 125 |
| ceiling | force ceiling 5 … 60 N through `translational_clip`, K = 2000 N/m | 175 |
| reference | the streaming chunk reference vs the goal held per control step | 50 |
| schedule | variable impedance: gains switched by task phase (free → search → slide → seat), 7 schedules over the error set and 15 random lock poses each | 280 |
| profile | time series for the force-vs-depth figure | 6 |

Phases for the schedules, from the oracle's state and the key's depth: `free` (approach and align), `search` (inserting,
depth < 3 mm: the chamfer), `slide` (3–24 mm: the blade in the slot), `seat` (≥ 24 mm: the plunger, and holding).

| schedule | free | search | slide | seat |
|---|---|---|---|---|
| fixed | robot's gains | same | same | same |
| soft_rot_insert | robot's | K_rot 15 | K_rot 15 | robot's |
| wrist_rot_insert | robot's | K_rot 1.5 (the soft wrist's tilt stiffness) | K_rot 1.5 | robot's |
| soft_rot_always | K_rot 15 | K_rot 15 | K_rot 15 | K_rot 15 |
| soft_insert | robot's | K 500, K_rot 15, clip 40 mm (20 N kept) | same | robot's |
| soft_insert_firm_seat | robot's | K 500, K_rot 15, clip 40 mm | same | clip 20 mm (40 N) |
| gentle_search | robot's | K_rot 15, clip 5 mm (10 N) | K_rot 15 | robot's |

## Results

All numbers are from `results/summary.md` (948 episodes, FR3 gains, `python -m arm_impedance.evaluate`).

![success maps](figures/success_maps.png)

**The robot's controller beats the sim's OSC on this task, and never exceeds 20 N.** Over the 9 × 9 grid the HSA
law seats 27 of 81 (33%) against OSC's 15 (19%), with a median peak flange force of 19.8 N against 43.3 N. Every OSC
failure is a jam at the oracle's 40 N stop; every HSA failure is a stall at the ceiling. Until this study the sim's
teleop and the key-insertion study ran on OSC, so their "the stiff wrist jams at 40 N" numbers partly measured OSC,
not the robot.

**The arm corrects lateral error completely and yaw error not at all.** With the robot's gains every dx up to ±2 mm
seats when the yaw error is ≤ 1.5°, and nothing seats at ≥ 3°, whatever dx. The boundary is between 2° and 3° (the
25-case set: all of ±2° seat, none of ±4°). OSC additionally fails for dx ≥ +0.5 mm. The lateral correction is the
arm's 2 kN/m spring yielding to the chamfer: 4 N of lateral force, 2 mm of travel.

![stiffness sweeps](figures/stiffness_sweeps.png)

**No arm gain changes the yaw limit.**

| sweep | finding |
|---|---|
| rotational stiffness 15 → 450 N·m/rad | identical: 60% (all ≤ 2°, none at 4°). At 5 N·m/rad lateral cases start to fail (36%); at 1.5 N·m/rad, the soft wrist's own tilt stiffness, the key droops 18° under the push and nothing seats, because the rotational clip (0.05 rad) caps the torque at 0.075 N·m. Turning a 4° yaw error through the blade's 7 mm half-width would need ~1 N·m from a 20 N push; the chamfer cannot produce it at any stiffness the arm can hold the tool with |
| translational stiffness, 20 N ceiling kept | 1000–4000 N/m identical (60%). 8000 N/m wedges more: 44%, 10 N lateral, 18 N push to seat. 500 N/m seats nothing: the arm lags 20 mm under the plunger's 10 N (load / K), past the oracle's 15 mm overrun, and the ~0.5 N nullspace leak already shifts the hover 1 mm sideways, which with a 1.5 mm error misses the 2 mm chamfer |
| force ceiling (clip), K = 2000 | 5 N: nothing seats, the plunger needs 4–10 N, and the yaw cases show 50–66 N peaks, from the rotational spring's 7.5 N·m torque ceiling acting through the key's 17 cm lever arm, not from the translational spring. 10 N: 36% (seating marginal, 9.7 N push). **15 N and above: identical to 20 N**, up to 60 N. Raising the clip buys nothing on this task |
| reference | chunks vs the goal held per step: identical success, 17.9 vs 16.8 N peak. At 20 Hz the interpolator doesn't change the outcome |

![schedules](figures/schedules.png)

**Variable impedance by phase does not raise success; one schedule lowers the force of a failure.** All seven
schedules seat the same 15 of 25 error cases and all 15 random lock poses. `gentle_search` (10 N ceiling and
K_rot 15 while the key is in the chamfer, the robot's gains afterwards) cuts the median peak flange force on the
yaw-failure cases from 19.8 to 10.8 N at no cost in success or time. Softer translation during insertion
(`soft_insert`, K 500) costs success (48%) for the lag reason above; `soft_rot_always` raises the seating push to
19.4 N because the tool's orientation is no longer held while the plunger is pressed.

![force profiles](figures/force_profiles.png)

With no error, all three controllers slide in at zero force and meet the plunger at 26 mm; the HSA law seats at the
plunger's own 10 → 13.5 N and OSC overshoots to 22 N. With 1 mm and 3° of error, OSC spikes to 39 N at 1.3 mm depth;
the HSA law stalls at 2 mm with 14 N (robot gains) or 20 N.

## What to change on the robot

1. **Keep the 20 N ceiling for key insertion; don't raise `translational_clip` for it.** 15 N is enough to seat, 10 N is
   marginal, and above 20 N nothing improves. The October "raise the clip first" expectation was for turning, not
   insertion, and the sim says insertion failures are alignment, which force cannot fix.
2. **Stop treating yaw as the arm's problem.** A rigidly held key tolerates about 2° of yaw with the robot's
   controller and no gain change extends that. Yaw has to come from perception (the policy) or from the wrist, whose
   tilt stiffness is 100× softer than anything the arm can run and still hold the tool.
3. **Gain schedule worth trying on the robot: a 10 N ceiling while searching, 20 N once the blade is in.** Same success,
   half the force when it fails. It only needs `translational_clip` switched live on a depth or contact cue, which the
   controller already supports (`ros2 param set`).
4. **Don't run the arm below ~1000 N/m with a demo-trained policy.** The tracking lag under a 10 N load is 10 mm at
   1000 N/m and 20 mm at 500, so the policy would have to command targets that far past the seat, which it never saw
   in demonstration; and the nullspace leak starts to move the tool by a millimetre.
5. **The FR3 and FER nullspace tiers differ** (0.2 / 100 vs 1.0 / 10) and the sim ran the FR3's. Re-run `rot_k` and
   `grid` with `HSA_ROBOT=fer` before comparing against FER sessions, and decide which tier the FR3 should run.

Sim caveats: the oracle lowers at a fixed 10 mm/s and never uses force to steer; the lock is rigid and the key 10 g;
the "wrist-less" rig is the contracted Stewart wrist, 50× stiffer than the arm but not infinitely so.

## Wrist stiffness by task phase

`wrist_study.py` runs the same expert with the wrist's stiffness changed during the task, with the robot's arm law
throughout. Two mechanisms: **extension**, the real one, where softening means extending the legs (and the plate
moves 20 mm per unit of z, so the arm's target is moved the other way by the plate's load-free travel, rate-limited to
one unit per second, with the expert pausing its descent meanwhile); and **scale**, an idealised stiffness change with
no plate motion, which the HSA wrist cannot do but which bounds what stiffness alone could buy. Nine schedules over
the same phases as the arm study, 27 error cases (dx ±1.5 mm × yaw up to ±8°) and 10 random lock poses each,
333 episodes. Results in `results_wrist/summary.md`.

![wrist schedules](figures/wrist_schedules.png)
![yaw tolerance](figures/wrist_yaw_tolerance.png)

| schedule | 0° | 2° | 4° | 6° | 8° | peak flange when misaligned (N, med) |
|---|---|---|---|---|---|---|
| fixed stiff / home | 3/3 | 6/6 | 0/6 | 0/6 | 0/6 | 20 |
| fixed soft | 3/3 | 6/6 | 6/6 | 6/6 | 5/6 | 20 |
| soft search, stiff seat | 3/3 | 6/6 | 0/6 | 4/6 | 5/6 | 92 |
| soft until seat, then stiff | 3/3 | 6/6 | 0/6 | 2/6 | 5/6 | 94 |
| soft search only, stiff slide | 3/3 | 6/6 | 0/6 | 0/6 | 0/6 | 97 |
| scale: soft during search and slide | 3/3 | 6/6 | 0/6 | 1/6 | 0/6 | 20 |
| scale: soft always | 3/3 | 6/6 | 6/6 | 1/6 | 0/6 | 20 |

**Fixed soft wins, and every schedule that stiffens while the key is in the slot loses.** The soft wrist seats 26 of
27 error cases up to 8° of yaw at the same 20 N the rigid mount stalls at. Stiffening at the seat or during the slide
costs success *and* multiplies the force: 92 to 97 N peaks against the 20 N ceiling. The mechanism is visible in the
traces: with a yaw error the soft wrist is twisted by that angle while the blade follows the slot, the misalignment
is stored elastically in the legs, and stiffening re-applies it. The twist stiffness goes from 1.6 to 417 N·m/rad,
a 4° twist becomes 29 N·m of torque on a key the slot will not let turn, and the arm's clip does not bound it because
it is the wrist's own actuators doing the pushing. The idealised scale switch, with no plate motion, fails the same
way for the same reason, at lower force because it is limited to the home wrist's stiffness range. The one schedule
that worked, 4 of 6 at 6°, worked only where the stored misalignment happened to be small.

What this says about tunable stiffness: for key insertion the compliance has to stay soft from the search through
seating, because the misalignment it absorbs never goes away while the blade is in the slot. Stiffness is wanted in
phases the sim does not model, turning and opening the lock, where torque has to pass through the wrist, and a switch
there would be made with the key seated and the slot constraining the blade, the exact situation the sim shows to be
dangerous. So the stiffness change needs either a short release of the constraint first, or a schedule whose stiff
setting stores no misalignment, which is a question for the calibrated wrist on hardware. The compensation the
allocator needs is also now specified: the arm must move by the plate's *load-free* travel (measured plate minus
force over the sysid stiffness), not the measured plate position, or a deflected wrist un-compressing as it
stiffens drives the tool into the work.

## Interim wrist calibration (`wrist_calibration.py`)

What the real wrist data gives today, and the preset built from it (`--wrist-calib measured_v0`):

| quantity | measured | source | sim default | preset |
|---|---|---|---|---|
| z stroke, 0 → 360° | 18.9 mm, ≤ 0.8 mm hysteresis | 11-08 sweep | 40 mm (±20) | ±19 mm |
| extended axial stiffness | 2.5–5.6 kN/m, little deflection below 10 N | wrist/v1 demos, 65k contact samples | 1.5 kN/m | 3.5 kN/m (leg 700 N/m) |
| extended lateral stiffness | ~2.1 kN/m | same | 140 N/m | 330 N/m, **6× too soft**: the strut model cannot match both |
| contracted, home | not measured | | 100 kN/m, 6 kN/m | unchanged, interpolated |

With the preset, the three-wrist comparison over ±8° of yaw (`results_small_wrists_measured_v0/`):

| wrist | seats at 2° | 4° | 6° | 8° |
|---|---|---|---|---|
| stiff, home | 6/6 | 0/6 | 0/6 | 0/6 |
| soft, mentor defaults | 6/6 | 6/6 | 6/6 | 5/6 |
| soft, measured_v0 | 6/6 | 6/6 | 0/6 | 0/6 |

The soft wrist's forgiveness drops from 8° to 4° once its axial stiffness is set from the demos, and the lateral number
says the real one is stiffer still. Treat every soft-wrist tolerance from the sim as an upper bound until the push test
(0 / 120 / 240 / 360°, axial and lateral) replaces the interpolation.

## 1-D stiffness: insertion with the wrist's z held at one level (`z_sweep.py`)

The wrist's z is the one axis the real wrist's stiffness follows: out toward the keyhole (extended, +1) is soft, in
toward the base plate (contracted, −1) is stiff. `z_sweep.py` holds it at one of nine levels through the whole
insertion, 15 aiming errors per level (dx ±1.5 mm × yaw 0–8°), robot's arm law, both wrist models. Each level is
labelled with the HSA twist the 11-08 sweep measured for that plate height. 270 episodes, `results_z/summary.md`.

![z sweep](figures/z_sweep.png)

| z | HSA twist | leg k, default | seats yaw ≤ 2° | 4° | 6° | 8° | peak flange | lateral force on the key |
|---|---|---|---|---|---|---|---|---|
| −1 to +0.5 | 0–226° | 100 kN/m → 1.4 kN/m | all | none | none | none | 20 N | 4.1 → 0.8 N |
| +0.75 | 278° | 640 N/m | all | all | none | none | 20 N | 0.7 N |
| +1 | 360° | 300 N/m | all | all | all | 2 of 3 | 13 N | 1.5 N |

**It is a threshold, not a slope.** Seven of the nine levels, from fully contracted to 226° of twist, give the same
result: every attempt within 2° of yaw seats, nothing past it, at the 20 N ceiling. The only thing that changes across
those seven levels is the lateral force the key takes, 4 N down to 0.8 N. Forgiveness appears only in the last quarter
of the stroke: at 278° the wrist seats 4° of yaw, and at 360° it seats 6° and most of 8°, with the peak force dropping
to 13 N because the wrist absorbs the plunger's push. With the measured calibration (leg 700 N/m at full extension)
the threshold moves to the very end: only 360° forgives anything, and only 4°.

What that means for a 1-D stiffness policy on insertion: there is no intermediate setting worth learning. The useful
range is the top 20% of the stroke, and the right answer for inserting is "as extended as the wrist goes". The
intermediate levels buy only a smaller lateral load on the key, which matters for a bent key, not for success. A 1-D
policy on this task would learn a constant, which matches the mentor's RL finding; the stiffness dimension only earns
its keep when a later phase needs rigidity (the door's stiff lock, previous section), and then the useful switch is
from 360° back to roughly home, not to fully contracted.

Caveat: the real wrist's lateral stiffness at 360° is ~6× the model's, so the real threshold is likely at or beyond
the end of the stroke; the push test at 0 / 120 / 240 / 360° is what pins it.

## The door: wrist schedules and what makes a real door hard

`door_study.py` runs the schedule study on `KeyDoor` (branch `feature/door_task`): insert, turn the key to the
cylinder's stop, push the door open, with the wrist fixed or switched by phase (free → search → slide → seat → turn →
push), over 15 aiming errors, on a normal lock (0.15 N m to turn) and a stiff one (0.4 N m), with the mentor's wrist
model and the interim calibration, plus sweeps of cylinder friction, slot friction, door mass and hinge friction.
490 episodes, 4 workers at low priority, checkpointed. Results in `results_door/summary.md`.

![door schedules](figures/door_schedules.png)

| setting | normal lock | stiff lock | turn moment (N m) | push force (N) |
|---|---|---|---|---|
| fixed stiff | 40% | 40% | — | — |
| fixed home | 60% | 60% | 0.7 | 15 |
| fixed soft | **100%** | **0%: cannot turn** | 0.4 | 13 |
| soft insert, stiff turn + push | 100% | 100% | 4.3 | 42 |
| soft insert, home turn + push | 100% | 100% | 1.9 | 19 |
| soft until push, stiff push | 100% | 0%: cannot turn | 0.4 | 36 |
| scale: soft insert, home turn | 100% | 100% | 1.3 | 19 |

**On the door, the switch earns its keep.** Fixed soft is still the best insertion setting (100% against 40% stiff),
but a soft wrist cannot turn a stiff lock: its 1.6 N m/rad of twist stiffness winds up and the cylinder never moves,
0 of 15 at 0.4 N m. Every schedule that stiffens *for the turn* opens the stiff door 15 of 15; the one that stays soft
through the turn fails exactly like fixed soft. That is the tunable-stiffness result the flat task could not show:
compliance to find the slot, rigidity to transmit the torque.

**But stiffen to home, not to fully stiff.** Switching to the contracted wrist for the turn costs 4.3 N m at the
flange and 42 N on the push, against 1.9 N m and 19 N for a switch to home, with the same success. The stiff wrist
re-applies the aiming error the soft wrist absorbed (the same mechanism as the flat task's 90 N), and home is stiff
enough to turn the lock. The idealised stiffness-only switch (no plate motion) does the same at 1.3 N m.

**The interim calibration changes little here.** With the measured axial stiffness the soft wrist still opens 15 of
15 and the switch to stiff costs 2.7 N m instead of 4.3, because the calibrated soft wrist stores less twist.

![door factors](figures/door_factors.png)

The factor sweeps, three fixed wrists at yaw 0 / 2 / 4°: cylinder friction is the only factor that changes who
succeeds (the soft wrist fails from 0.4 N m); slot friction raises the insertion force on the stiff and home wrists
(12 → 20 N) and not on the soft one; door mass from 1.5 to 10 lb changes nothing, because the push is a force command
through the clip and the hinge is damped; hinge friction up to 2 N m raises the soft wrist's push force from 12 to
20 N and the stiff wrist's from 30 to 40 N, with no failures.

Two modelling notes. MuJoCo's joint `frictionloss` is a soft constraint scaled by the joint's inertia, and a 50 g
lock cylinder is too light for it to hold more than ~0.1 N m whatever is set; `KeyDoor` applies the cylinder's
friction as an explicit Coulomb torque instead, and gives the cylinder enough armature for that to be stable. And a
turn loads the flange with a *moment*, so the ledger carries the peak flange moment per phase next to the force; the
force alone hides the lock entirely.

## A learned stiffness head (`stiffness_head.py`)

A 2-layer MLP that outputs the wrist's z from the robot's observations, trained on the 200 scheduled-expert door
demos (`key_insertion/door_demos.py`, 160 train / 40 validation episodes) and evaluated closed-loop in the sim: the
expert still aims the key, the head's z replaces the schedule and drives the wrist through the same compensated
switching. Two input sets, a 5-step window at 20 Hz: **proprio** (tcp pose, tool pose, flange wrench) and **vision**
(proprio + the key tip's pose in the keyhole frame, standing in for a camera). Target: the expert's z 0.25 s ahead.

| head | val MAE (z in 0..1) | normal lock | stiff lock | z during insert | z during turn | turn moment | push |
|---|---|---|---|---|---|---|---|
| expert schedule (soft insert, home turn) | — | 15/15 | 15/15 | 1.0 | 0.0 | 1.9 N m | 19 N |
| proprio | 0.009 | 14/15 | 13/15 | 0.96 | 0.16 | 1.9 N m | 19 N |
| vision | 0.013 | 13/15 | 13/15 | 0.97 | 0.15 | 2.0 N m | 19 N |
| fixed soft, for reference | — | 15/15 | 0/15 | 1 | 1 | 0.4 N m | 13 N |

Both heads learn the mechanism from demonstration: they soften for the insertion and stiffen to home after seating,
on their own initiative, with the schedule's forces. The failures are all at +4° of yaw, outside the ±3° the demos
were generated with; at ±2° and −4° every episode opens. Proprioception alone is enough: the wrench and the tool's
motion mark seating well enough that the lock pose adds nothing.

Two things the first attempt taught, both of which carry to hardware:

- **Keep the wrist's own state out of the stiffness head's inputs.** With the measured plate pose available, the head
  reached a validation error of 0.006 by reading the current stiffness off the wrist, and closed-loop it never
  initiated a switch: 0 switches in 30 episodes, home wrist throughout, 9/15 on both locks. The robot's
  `robot0_wrist_ext_deg` / plate pose streams are the same leak.
- **Predict ahead.** The target is z a quarter second ahead, so the head has to anticipate the switch from the
  approach and the contact, not report it.

Weights are plain `.npz` (`heads/head_*.npz`, with the input list and normalisation), so the sim side runs them in
numpy; training needs torch (`~/hsa-ws/eval_venv`). Regenerate the demos and retrain once the push test calibrates
the wrist.

## Small runs, the ledger, and hardware

The big study is many episodes of the same thing. To see what one episode is, run the small version: a few
perception errors, a few controllers, one printed line per episode, then the ledger and two charts:

```bash
cd scripts && source ../robosuite_env/bin/activate
python -m arm_impedance.small_eval                                    # 30 episodes: robot law and OSC, 3 dx x 5 dyaw -> results_small/
python -m arm_impedance.small_eval --controllers hsa --wrist soft --out arm_impedance/results_small_soft   # with the wrist
python -m arm_impedance.small_eval --schedule gentle_search --clip 5  # a schedule, or a different ceiling
```

`results_small/episodes.csv` is the **ledger**: one row per attempt with `label`, `dx_mm`, `dyaw_deg`, `result`
(seated / stuck / jammed / timeout), `peak_flange_N`, and whatever else was measured. `ledger_figures.py` draws the
success map and the force / time bars from any ledger with those columns, which is the point: a hardware session
logs the same rows by hand (or from the eval runner), and the same two charts come out of

```bash
python -m arm_impedance.ledger_figures path/to/hardware/episodes.csv --out path/to/figs
```

`hardware_ledger_example/` is a filled-in example. On the robot, `dx_mm` / `dyaw_deg` are how far the lock fixture
was moved from where the policy (or the operator) expects it, `result` is the operator's call, and `peak_flange_N`
comes from the Franka's external wrench (`O_F_ext_hat_K`, zeroed in free air). With the wrist back, the same ledger
takes a `wrist` column (stiff / home / soft, or the servo setpoint), and `small_eval --wrist` already produces it.

The sim and hardware ledgers are deliberately the same shape so that a sim prediction (for example "2° of yaw is the
limit, at any gain") can be checked against the robot with the same chart.

**The real ledgers HSA-UMI already has** (the demo recorder's and `rollout_ledger`'s CSVs under
`HSA-UMI/data/key_in_hole`: 302 teleop episodes over rigid v1 / v2 and wrist v1, and 55 policy rollouts) convert
into the same ledger and the same charts with

```bash
python -m arm_impedance.hsa_ledger ../../HSA-UMI/data/key_in_hole --out arm_impedance/hardware_ledgers
```

`hardware_ledgers/` and `hardware_ledgers_demos/` hold that output. Those ledgers carry outcome, condition and
timing but no placement offset (every rollout so far was at the nominal pose), so the success map collapses to one
cell; the success bars are the comparison.

**The force is in the bags.** Every rollout session records `/franka_robot_state_broadcaster/external_wrench_in_base_frame`
(libfranka's external-wrench estimate) in one rosbag per session, and the ledger's `started_at` / `time_s` cut each
trial out of it. `bag_forces.py` does that, zeroes each trial with the 2 s of stillness before it, and writes
`peak_flange_N` into the ledger and the charts (needs `rosbags`, in requirements.txt):

```bash
python -m arm_impedance.bag_forces ../../HSA-UMI/data/key_in_hole/rigid/v2/eval/v2f_best --out arm_impedance/hardware_ledgers/v2f_best
```

`hardware_ledgers/<session>/` hold those. Read them with care: the window is the whole trial (insert, turn, open),
not just the insertion, and the Franka's estimate carries the payload model's error, so they are not yet the same
quantity as the sim's insertion-only peak. The sim's 20 N ceiling bounds what the controller *commands*; the
estimate also sees what the lock pushes back with while the key is turned.

**Videos.** `small_eval --video` saves an mp4 of every episode under `<out>/videos/`, a close-up of the lock with the
phase, depth and push overlaid, and the outcome stamped on the last second.

## Reproducing everything

```bash
cd scripts && source ../robosuite_env/bin/activate
python -m pytest arm_impedance/test_arm_impedance.py -q    # the harness, ~30 s
python -m arm_impedance.evaluate                           # 948 episodes, ~5 min on 20 cores -> results/
python -m arm_impedance.make_figures                       # -> figures/*.png
python -m key_insertion.oracle --arm hsa --dx 1 --dyaw 3 --wrist stiff   # watch one case (python, not mjpython, on Linux)
```

`results/episodes.csv` has every episode (gains, error, outcome, peak flange and lateral force, push while seated,
contact-to-seat time, max yaw of the key); `results/summary.md` the tables; `results/profiles.npz` the time series.
