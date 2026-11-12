"""
Wrist stiffness scheduled by task phase, on KeyInsertion with the robot's arm controller: does softening the wrist
while the key hunts for the slot and stiffening it to seat beat any fixed wrist setting, and how large an aiming
error does each forgive?

    cd scripts
    python -m arm_impedance.wrist_study            # ~330 episodes, ~2 min on 20 cores -> arm_impedance/results_wrist/
    python -m arm_impedance.wrist_study --quick
    python -m arm_impedance.wrist_figures          # -> arm_impedance/figures/wrist_*.png

Two ways to change the wrist's stiffness during a task, both run:

  extension   the real mechanism. The HSA legs' stiffness follows their extension (300 N/m extended, 100 kN/m
              contracted), so softening = extending. Extending moves the plate, and the key with it, by 20 mm per
              unit of z, so the arm's target is moved the opposite way in lockstep (what the wrist allocator on the
              robot does), and the change is rate-limited (`RATE` per second of normalized z) so the key is never
              yanked, and the expert pauses its descent while the wrist changes (a policy would hold too). The
              residual tool motion during a switch, tip relative to its target, is recorded (`switch_tool_mm`).
  scale       an idealized switch: every leg's stiffness is multiplied by a factor with the plate held still
              (`StewartPlatformWrist.stiffness_scale`). Not something the HSA wrist can do; it is the upper bound on
              what a stiffness change alone, with no motion, could buy.

Phases from the key's depth (as in evaluate.py): free -> search (< 3 mm) -> slide (3-24 mm) -> seat (>= 24 mm).
The error set is wider in yaw than the arm study's, because the soft wrist forgave all of that one: dx +-1.5 mm,
dyaw up to +-8 deg. Random lock poses (no error) check that nothing regresses on an honest attempt.
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arm_impedance.evaluate as ae  # noqa: E402
from arm_impedance.evaluate import task_phase  # noqa: E402
from key_insertion.oracle import KeyOracle, make_T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_wrist")
STIFF, HOME, SOFT = -1.0, 0.0, 1.0
RATE = 1.0  # normalized z per second: a full soft -> stiff switch takes 2 s, 40 mm of plate travel at 20 mm/s
SOFT_SCALE = 0.05  # home wrist (6 kN/m legs) x 0.05 = 300 N/m, the extended wrist's stiffness, with no plate motion

# phase -> (wrist z, stiffness scale)
SCHEDULES = {
    "fixed_stiff": {p: (STIFF, 1.0) for p in ("free", "search", "slide", "seat")},
    "fixed_home": {p: (HOME, 1.0) for p in ("free", "search", "slide", "seat")},
    "fixed_soft": {p: (SOFT, 1.0) for p in ("free", "search", "slide", "seat")},
    "soft_search_stiff_seat": {"free": (STIFF, 1.0), "search": (SOFT, 1.0), "slide": (SOFT, 1.0), "seat": (STIFF, 1.0)},
    "soft_search_home_seat": {"free": (HOME, 1.0), "search": (SOFT, 1.0), "slide": (SOFT, 1.0), "seat": (HOME, 1.0)},
    "soft_until_seat": {"free": (SOFT, 1.0), "search": (SOFT, 1.0), "slide": (SOFT, 1.0), "seat": (STIFF, 1.0)},
    "soft_search_only": {"free": (STIFF, 1.0), "search": (SOFT, 1.0), "slide": (STIFF, 1.0), "seat": (STIFF, 1.0)},
    "scale_soft_search": {"free": (HOME, 1.0), "search": (HOME, SOFT_SCALE), "slide": (HOME, SOFT_SCALE), "seat": (HOME, 1.0)},
    "scale_soft_always": {p: (HOME, SOFT_SCALE) for p in ("free", "search", "slide", "seat")},
}  # fmt: skip
MECHANISM = {n: ("scale" if n.startswith("scale") else "extension" if not n.startswith("fixed") else "fixed") for n in SCHEDULES}

ERRORS = [(dx, 0.0, dyaw) for dx in (-1.5, 0.0, 1.5) for dyaw in (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)]


PHASE_ORDER = ("free", "search", "slide", "seat")
SCALE_RAMP_S = 1.0  # a stiffness-scale change is spread over this long (log-linear): a snap while deflected wedges the key


class WristScheduler:
    """
    on_step hook: moves the oracle's wrist command toward the phase's setting. Phases only advance (a key that bounces
    at the seat boundary must not flip the wrist back and forth). During an extension switch the arm's target is
    moved by the plate's *measured* displacement along the tool axis, so the key stays put while the legs change
    length; outside a switch nothing is compensated, so the wrist's compliance is exactly the fixed runs'.
    """

    def __init__(self, name):
        self.schedule = SCHEDULES[name]
        self.current = None
        self.target_z = None
        self.switching = False
        self.switch_tool_mm = 0.0  # largest unintended key-tip motion along the keyhole axis during a switch (tip vs target)
        self.n_switches = 0
        self.scale_from = self.scale_to = None
        self.scale_t0 = None

    def __call__(self, env, oracle, row):
        phase = task_phase(oracle, row["depth"])
        if self.current is None or PHASE_ORDER.index(phase) > PHASE_ORDER.index(self.current):
            z, scale = self.schedule[phase]
            self.current = phase
            if abs(scale - oracle.wrist.stiffness_scale) > 1e-9:
                self.scale_from, self.scale_to, self.scale_t0 = oracle.wrist.stiffness_scale, scale, row["t"]
            if abs(z - oracle.wrist_z) > 1e-9 and not self.switching:
                self.n_switches += 1
                self.switching = True
                self.target_z = z
                self._T0 = oracle.T_tip_tcp.copy()
                self._plate0 = self._free_plate_z(oracle, row)
                self._err0 = self._tip_axis_pos(oracle) + oracle.insert_height  # tip vs its target, before the switch
                self._height0 = oracle.insert_height  # the descent pauses while the wrist changes
                self._settled_since = None
        if self.scale_t0 is not None:
            a = min(1.0, (row["t"] - self.scale_t0) / SCALE_RAMP_S)
            oracle.wrist.stiffness_scale = float(np.exp((1 - a) * np.log(self.scale_from) + a * np.log(self.scale_to)))
            if a >= 1.0:
                self.scale_t0 = None
        if self.switching:
            step = np.clip(self.target_z - oracle.wrist_z, -RATE * oracle.dt, RATE * oracle.dt)
            oracle.wrist_z += step
            # the plate carries the key: move the arm's tcp target by the plate's travel, the other way. The travel
            # is the measured plate z minus its load deflection (force / axial stiffness), so that a soft wrist
            # un-compressing as it stiffens under the plunger isn't mistaken for commanded motion
            d = self._free_plate_z(oracle, row) - self._plate0
            oracle.T_tip_tcp = self._T0 @ make_T(np.eye(3), [0.0, 0.0, -d])
            if oracle.phase in (KeyOracle.INSERT, KeyOracle.HOLD):
                oracle.insert_height = self._height0  # a stiffness change is a pause: the expert holds its target
            err = self._tip_axis_pos(oracle) + oracle.insert_height
            self.switch_tool_mm = max(self.switch_tool_mm, abs(err - self._err0) * 1e3)
            at_target = abs(self.target_z - oracle.wrist_z) < 1e-9
            if at_target:
                self._settled_since = row["t"] if self._settled_since is None else self._settled_since
                if row["t"] - self._settled_since > 0.3:
                    self.switching = False  # freeze the compensation where the plate ended up
        return True

    @staticmethod
    def _free_plate_z(oracle, row):
        """measured plate z (m) with the load deflection removed: where the plate would be with no force on it"""
        k_zz = oracle.wrist.platform_stiffness(oracle.wrist.last_commanded_pose)[2, 2]
        return row["wrist_pose"][2] - row["flange_force"][2] / k_zz

    @staticmethod
    def _tip_axis_pos(oracle):
        """key tip position along the believed keyhole axis (m): what a wrist switch must not move"""
        T = oracle.site_T(oracle.gripper.important_sites["grip_site"])
        L = oracle.believed_lock_T()
        return float((np.linalg.inv(L) @ T)[2, 3])

    def summary(self):
        return {"n_switches": self.n_switches, "switch_tool_mm": round(self.switch_tool_mm, 2)}


def cases(quick=False):
    out = []
    errors = ERRORS if not quick else [e for e in ERRORS if e[0] == 0.0]
    for name, sched in SCHEDULES.items():
        z0 = sched["free"][0]
        for dx, dy, dyaw in errors:
            out.append(ae._case("wrist", name, overrides=ae.gains(), dx=dx, dy=dy, dyaw=dyaw, wrist_z=z0, wrist_schedule=name, mechanism=MECHANISM[name], max_seconds=25.0))
        for seed in range(3 if quick else 10):
            out.append(ae._case("wrist", name, overrides=ae.gains(), wrist_z=z0, wrist_schedule=name, mechanism=MECHANISM[name], fixed_lock=False, seed=seed, max_seconds=25.0))
    return out


COLUMNS = "| schedule | mechanism | episodes | seated | stuck | timeout | peak flange (N, med) | peak lateral (N, med) | push seated (N, med) | contact→seat (s, med) | switch tool motion (mm, max) |"


def _table(rows):
    lines = [COLUMNS, "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label in SCHEDULES:
        w = [r for r in rows if r["label"] == label]
        if not w:
            continue
        n = lambda res: sum(r["result"] == res for r in w)  # noqa: E731
        push = [r["hold_push_N"] for r in w if r["success"]]
        seat = [r["contact_to_seat_s"] for r in w if r["success"] and np.isfinite(r["contact_to_seat_s"])]
        lines.append(
            f"| {label} | {MECHANISM[label]} | {len(w)} | {n('seated')} ({100 * n('seated') / len(w):.0f}%) | {n('stuck')} | {n('timeout')} | "
            f"{np.median([r['peak_flange_N'] for r in w]):.1f} | {np.median([r['peak_lateral_N'] for r in w]):.1f} | "
            f"{np.median(push) if push else float('nan'):.1f} | {np.median(seat) if seat else float('nan'):.2f} | "
            f"{max(r.get('switch_tool_mm', 0.0) for r in w):.1f} |"
        )
    return lines


def write_summary(rows, path):
    lines = ["# Wrist stiffness by task phase: KeyInsertion oracle results", "", "Generated by `python -m arm_impedance.wrist_study`. Robot's arm law and FR3 gains throughout.", ""]
    err = [r for r in rows if r["fixed_lock"] and r["result"] != "error"]
    rnd = [r for r in rows if not r["fixed_lock"] and r["result"] != "error"]
    lines += ["## 27 error cases (dx ±1.5 mm × dyaw ±8°)", ""] + _table(err) + [""]
    lines += ["## largest yaw error seated, per schedule (dx = 0)", "", "| schedule | seats up to | fails from |", "|---|---|---|"]
    for label in SCHEDULES:
        w = [r for r in err if r["label"] == label and r["dx_mm"] == 0.0]
        ok = [abs(r["dyaw_deg"]) for r in w if r["success"]]
        bad = [abs(r["dyaw_deg"]) for r in w if not r["success"]]
        lines.append(f"| {label} | {max(ok) if ok else 0:.0f}° | {min(bad) if bad else float('nan'):.0f}° |")
    lines += ["", "## random lock poses, no error", ""] + _table(rnd) + [""]
    errors = [r for r in rows if r["result"] == "error"]
    if errors:
        lines += [f"{len(errors)} episodes raised an error (see episodes.csv)", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    todo = cases(args.quick)
    print(f"{len(todo)} episodes on {args.workers} workers")
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(ae.run_case, todo, chunksize=2)):
            rows.append(row)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    order = {n: i for i, n in enumerate(SCHEDULES)}
    rows.sort(key=lambda r: (order[r["label"]], r["dx_mm"], r["dyaw_deg"], r["seed"]))
    os.makedirs(RESULTS, exist_ok=True)
    fields = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in ("keep_log", "overrides", "log")]
    with open(os.path.join(RESULTS, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    write_summary(rows, os.path.join(RESULTS, "summary.md"))
    print(open(os.path.join(RESULTS, "summary.md")).read())


if __name__ == "__main__":
    main()
