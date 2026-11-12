"""
A small, readable evaluation: a few perception errors x a few controllers, one line per episode, a ledger CSV and
two charts. The same ledger columns as the big study, so the charts come from `ledger_figures` and work on any
ledger, including one filled by hand from hardware trials.

    cd scripts
    python -m arm_impedance.small_eval                         # 30 episodes: robot law and OSC, 3 dx x 5 dyaw
    python -m arm_impedance.small_eval --controllers hsa --wrist stiff home soft   # the robot law with three wrists, one panel each
    python -m arm_impedance.small_eval --video                 # the same 30, with an mp4 of each under results_small/videos/
    python -m arm_impedance.small_eval --schedule gentle_search --dx 0 --dyaw -4 0 4
    python -m arm_impedance.small_eval --out results_small_fer --robot fer   (set HSA_ROBOT=fer first)

Writes <out>/episodes.csv and <out>/summary.md, and <out>/success_map.png + <out>/forces.png via ledger_figures.
"""

import argparse
import csv
import os
import sys
import time

if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering, before mujoco is imported

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arm_impedance.evaluate as ae  # noqa: E402
from arm_impedance.ledger_figures import LEDGER_COLUMNS, make_ledger_figures, summarize  # noqa: E402
from key_insertion.oracle import WRIST_PRESETS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"hsa": "robot law", "osc": "OSC"}
FPS = 10  # every other control step at 20 Hz


class VideoRecorder:
    """on_step hook: renders a close-up of the lock every other control step into an mp4 (OpenCV, no ffmpeg needed)"""

    def __init__(self, path, title):
        import cv2
        import mujoco

        self.cv2, self.mujoco, self.path, self.title = cv2, mujoco, path, title
        self.renderer = self.writer = None
        self.n = 0

    def __call__(self, env, oracle, row):
        self.n += 1
        if self.n % 2:
            return True
        m, d = env.sim.model._model, env.sim.data._data
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(m, 480, 640)
            self.cam = self.mujoco.MjvCamera()
            self.cam.lookat[:] = d.site_xpos[m.site("lock_site").id] + [0, 0, 0.04]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.32, 160, -12
            self.opt = self.mujoco.MjvOption()
            self.opt.geomgroup[0] = 0
            self.writer = self.cv2.VideoWriter(self.path, self.cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
        self.renderer.update_scene(d, self.cam, self.opt)
        frame = self.renderer.render()[:, :, ::-1].copy()  # RGB -> BGR
        push = -row["flange_force"][2]
        lines = [self.title, f"t {row['t']:5.1f} s  phase {oracle.PHASES[oracle.phase]}  depth {row['depth'] * 1e3:5.1f} mm  push {push:5.1f} N"]
        for i, text in enumerate(lines):
            self.cv2.putText(frame, text, (10, 24 + 22 * i), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, self.cv2.LINE_AA)
            self.cv2.putText(frame, text, (10, 24 + 22 * i), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, self.cv2.LINE_AA)
        self.writer.write(frame)
        return True

    def close(self, result):
        if self.writer is not None:
            frame = self.renderer.render()[:, :, ::-1].copy()
            self.cv2.putText(frame, result.upper(), (10, 460), self.cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, self.cv2.LINE_AA)
            self.cv2.putText(frame, result.upper(), (10, 460), self.cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, self.cv2.LINE_AA)
            for _ in range(FPS):  # hold the last frame a second
                self.writer.write(frame)
            self.writer.release()
            self.renderer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controllers", default="hsa,osc", help="comma list of hsa (the robot's law and gains), osc")
    p.add_argument("--dx", type=float, nargs="+", default=[-1.5, 0.0, 1.5], help="perception error across the slot (mm)")
    p.add_argument("--dyaw", type=float, nargs="+", default=[-4, -2, 0, 2, 4], help="perception error in yaw (deg)")
    p.add_argument("--wrist", nargs="+", default=["stiff"], help="one or more of stiff (contracted: the wrist-less stand-in), home, soft, or a z in [-1, 1]; each gets its own label")
    p.add_argument("--schedule", default="", choices=[""] + list(ae.SCHEDULES), help="hsa only: a variable-impedance schedule")
    p.add_argument("--clip", type=float, default=None, help="hsa only: translational_clip (mm), the force ceiling / 2000")
    p.add_argument("--seeds", type=int, default=1, help="repeat each case with this many random lock poses (seed 0 = fixed lock)")
    p.add_argument("--out", default=os.path.join(HERE, "results_small"))
    p.add_argument("--video", action="store_true", help="save an mp4 of every episode under <out>/videos/")
    args = p.parse_args()
    if args.video:
        os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)

    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    overrides = ae.gains(clip_t=args.clip * 1e-3) if args.clip is not None else ae.gains()
    rows = []
    print(f"{'#':>3} {'label':22s} {'believed error':>17s} | {'result':7s} {'depth':>8s} {'peak push':>10s} {'in contact':>10s}")
    t0 = time.time()
    n = 0
    for wrist in args.wrist:
        ae.WRIST_Z = WRIST_PRESETS[wrist] if wrist in WRIST_PRESETS else float(wrist)
        for arm in controllers:
            for seed in range(args.seeds):
                for dx in args.dx:
                    for dyaw in args.dyaw:
                        n += 1
                        label = NAMES[arm] + (f" {args.schedule}" if args.schedule and arm == "hsa" else "")
                        if len(args.wrist) > 1:
                            label += f", {wrist} wrist"
                        case = ae._case(
                            "small", label, arm=arm, overrides=overrides if arm == "hsa" else None, dx=dx, dyaw=dyaw,
                            schedule=args.schedule if arm == "hsa" else "", fixed_lock=seed == 0, seed=seed,
                        )  # fmt: skip
                        rec = None
                        if args.video:
                            name = f"{n:02d}_{label.replace(' ', '_').replace(',', '')}_dx{dx:+.1f}_dyaw{dyaw:+.0f}.mp4"
                            rec = VideoRecorder(os.path.join(args.out, "videos", name), f"{label}  believed error dx {dx:+.1f} mm, yaw {dyaw:+.0f} deg")
                        r = ae.run_case(case, on_step=rec)
                        if rec is not None:
                            rec.close(r["result"])
                            r["notes"] = f"video: videos/{name}"
                        r["wrist"] = wrist
                        r["source"] = "sim"
                        rows.append(r)
                        depth = r.get("depth_mm", float("nan"))
                        print(
                            f"{n:3d} {label:22s} {dx:+5.1f} mm, {dyaw:+3.0f} deg | {r['result']:7s} {depth:5.1f} mm "
                            f"{r.get('peak_flange_N', float('nan')):7.1f} N {r.get('contact_to_seat_s', float('nan')):7.1f} s"
                        )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summarize(rows, title=f"small eval: wrist {' / '.join(args.wrist)}, {len(rows)} episodes in {time.time() - t0:.0f} s"))
    print(open(os.path.join(args.out, "summary.md")).read())
    make_ledger_figures(os.path.join(args.out, "episodes.csv"), args.out)


if __name__ == "__main__":
    main()
