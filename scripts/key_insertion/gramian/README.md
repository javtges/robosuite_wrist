# Controllability Gramians vs wrist stiffness strategy

Do the controllability Gramian and its log-determinant differ between the rigid (z = -1), most flexible (z = +1) and
variable wrist, over whole episodes and per subtask phase? These scripts answer that for the SERL key-in-lock task and
the KeyDoor scheduled expert. They import the existing task code and change none of it.

| script | venv | what it does |
|---|---|---|
| `record_rollouts.py` | `~/serl_env` | rolls out `../runs/<mode>_seed0/final` with SAC's **sampled** actions (nominal + aim-error starts), logs per-step state/action |
| `gramian_lib.py` | either | phases, ridge local-linear models, Gramian / log-det helpers |
| `analyze.py` | either | key-in-lock: closed-loop identifiability, held-out R², per-phase LTI Gramians (episode bootstrap), kNN time-varying Gramians, outcome correlations, figures |
| `sim_impulse.py` | `~/serl_env` | key-in-lock simulator check: +-eps action impulses replayed open-loop, to validate the data-driven models |
| `door_gramian.py` | `robosuite_env` | KeyDoor: runs `door_demos.record_episode` unchanged for `fixed_stiff` / `fixed_soft` / `soft_insert_home_turn`, forks the process at every 2nd step and measures impulse responses (simulator / empirical Gramian) |
| `door_analyze.py` | `robosuite_env` | KeyDoor: windowed (1 s) Gramians per phase, outcome correlations, the data-only attempt for comparison, figures |

Run from `robosuite_wrist/scripts/key_insertion/gramian`:

```
~/serl_env/bin/python record_rollouts.py --out data/rollouts
~/serl_env/bin/python sim_impulse.py --rollouts data/rollouts --out data/impulse
python analyze.py --rollouts data/rollouts --out results/key_in_lock --impulse data/impulse/impulse.npy
python door_gramian.py --out data/door --episodes 8 --N 20 --stride 2   # ~1-2 h on 9 workers
python door_analyze.py --door data/door --out results/door
```

`data/` (raw rollouts and impulse responses, ~95 MB) is git-ignored; `results/` holds the figures and JSON reports.

## Findings (2026-11-18)

- **Learning the Gramian from trajectories** only works when the actions carry independent noise. With SAC's sampled
  actions the local models are identifiable, and the simulator agrees on the ordering. The door expert is
  deterministic: the state explains 98-99.9 % of its arm commands, and the data-only Gramians miss by 30-65 nats.
- **Alternatives:** stochastic rollouts with local linear models, or the simulator's empirical Gramian (`door_gramian.py`).
- **Stiffness vs Gramian:** in contact, wrench (and door/cylinder) authority orders rigid > variable >= flexible on
  both tasks; free space is the same for all three. The variable door schedule matches flexible through insertion and
  approaches rigid at the turn (door+cylinder block -19 vs rigid -18, flexible -26 nats), which is the soft wrist's
  "cannot turn" failure. Whole-episode log-det hides this: only the per-phase values separate the strategies. The
  Gramian tracks the contact state (high when wedged) rather than predicting success; 8 door episodes per schedule.
- Door "search" uses `door_study.door_phase` (depth < 3 mm), so it includes the approach above the lock. A
  door + cylinder log-det of -55.3 is the numerical floor: no authority yet.
