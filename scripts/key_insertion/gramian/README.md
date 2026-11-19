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
| `policy_entropy.py` | `~/serl_env` | exact SAC policy action entropy H(pi(.|o)) at every logged key-in-lock observation (from the checkpoints) |
| `entropy.py` | `robosuite_env` | state-action entropy H(x,u), H(x), H(u) (k-NN, equal n per group) and H(pi), same phases / conditions as the Gramian figures, both tasks |

Run from `robosuite_wrist/scripts/key_insertion/gramian`:

```
~/serl_env/bin/python record_rollouts.py --out data/rollouts
~/serl_env/bin/python sim_impulse.py --rollouts data/rollouts --out data/impulse
python analyze.py --rollouts data/rollouts --out results/key_in_lock --impulse data/impulse/impulse.npy
python door_gramian.py --out data/door --episodes 8 --N 20 --stride 2   # ~1-2 h on 9 workers
python door_analyze.py --door data/door --out results/door
~/serl_env/bin/python policy_entropy.py   # writes data/rollouts/policy_entropy_<mode>.npz
python entropy.py --draws 100            # results/{key_in_lock,door}/*entropy*
```

`data/` (raw rollouts and impulse responses, ~95 MB) is git-ignored; `results/` holds the figures and JSON reports.

## Saved Gramian matrices (`results/*/gramians.npz`, git-ignored, 27 MB + 8 MB)

Every Gramian behind the reported log-dets, in float64. Recomputing the log-dets from these files gives the reports exactly.
States are in normalized obs units (1 cm, 0.1 rad, 5 cm/s, 2 N m, 20 N, door / cylinder 0.1 rad); `state_names` gives the order.

Key-in-lock (`results/key_in_lock/gramians.npz`; `N` = horizon, 10 steps = 1 s; 15 states, or 16 with z for variable):

| key | shape | meaning |
|---|---|---|
| `<mode>/lti/<group>/W` | (n, n) | per-phase LTI Gramian, 4 motion inputs; group = all, free, search, slide, latch |
| `<mode>/lti/<group>/A`, `B` | (n, n), (n, m) | the local linear model it came from |
| `<mode>/lti/<group>/W_boot` | (200, n, n) | episode-bootstrap Gramians (the CIs) |
| `variable/lti/<group>/W5` | (16, 16) | same with z as a 5th input |
| `<mode>/ltv/window/W` | (T, n, n) | kNN time-varying Gramian of the 1 s window starting at each transition (nan near episode ends) |
| `<mode>/ltv/window/{ep,t,phase}` | (T,) | which episode / step / phase each window starts in |
| `variable/ltv/window/W5` | (T, 16, 16) | windows with z as an input |
| `<mode>/ltv/episode/W`, `ep` | (E, n, n), (E,) | whole-episode Gramian per episode |

Door (`results/door/gramians.npz`; 17 states, 1 s windows; `input_names`, `phases`):

| key | shape | meaning |
|---|---|---|
| `<mode>/W` | (K, 17, 17) | simulator Gramian of each window, 6 arm inputs |
| `<mode>/W7` | (K, 17, 17) | same with the wrist z command as a 7th input |
| `<mode>/{ep,n,phase,z}` | (K,) | episode, start step, phase index into `phases` (-1 done), wrist z at the window start |

```python
g = np.load("results/door/gramians.npz")
W = g["variable/W"][g["variable/phase"] == list(g["phases"]).index("turn")]
```

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
