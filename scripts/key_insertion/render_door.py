"""
GIFs of the KeyDoor scripted expert, the episodes door_demos.py records (modelled on render_scene.py):

    cd scripts && python -m key_insertion.render_door

figures/door_demo.gif   one door_demos.py episode: the default schedule soft_insert_home_turn (wrist soft, z = +1, to find
                        the slot and seat; home, z = 0, to turn and push) on the stiff lock (0.4 N m), 1 mm / 2 deg aim
                        error. Wide view and a side close-up of the wrist; wrist z, the moment on the arm's flange, and
                        the cylinder and door angles traced underneath with the phases marked
figures/door_turn.gif   the same lock and aim error from contact on, side by side: the schedule vs the wrist soft
                        throughout. The soft wrist seats the key but cannot turn it: its legs twist and the cylinder
                        stalls near 30 deg

Rendering only: two unused world cameras (sideview, agentview) are re-aimed at the lock after the reset; the model,
the oracle and the schedules are door_study.run_case's.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import robosuite.utils.transform_utils as T  # noqa: E402
from robosuite.environments.manipulation.key_insertion import _look_at  # noqa: E402

from arm_impedance.door_study import FIXED, PHASES, SCHEDULES, door_phase  # noqa: E402
from arm_impedance.wrist_study import WristScheduler  # noqa: E402
from key_insertion.door_oracle import DoorOracle, make_door_env, run_door_episode  # noqa: E402
from key_insertion.render_scene import AXIS, FIGS, INK, MUTED, SURFACE, font, render, trace  # noqa: E402

AIM = (1e-3, 0.0, np.radians(2.0))  # dx (m), dy, dyaw (rad): inside door_demos.py's +-1.5 mm / +-3 deg
LOCK = 0.4  # cylinder friction (N m): the stiff lock, where the soft wrist alone cannot turn the key
COLORS = {"soft_insert_home_turn": (42, 120, 214), "fixed_soft": (235, 104, 52)}
TITLES = {"soft_insert_home_turn": "soft to insert, home to turn", "fixed_soft": "soft throughout (z = +1)"}
PHASE_FILL = {"free": (246, 246, 244), "search": (236, 242, 250), "slide": (226, 236, 248), "seat": (214, 228, 246),
              "turn": (250, 236, 226), "push": (244, 226, 214)}  # fmt: skip
WIDE, CLOSE = "agentview", "sideview"


def aim_cameras(env):
    """Re-aim two world cameras at the keyhole: a close side view of the wrist from the latch edge, and a wide view"""
    m, d = env.sim.model, env.sim.data
    sid = m.site_name2id("lock_site")
    p, n = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3)[:, 2].copy()  # n: keyhole axis, out toward the robot
    side, up = np.cross([0, 0, 1.0], n), np.array([0, 0, 1.0])
    side /= np.linalg.norm(side)
    for cam, eye, target in ((CLOSE, p + 0.30 * side + 0.10 * n + 0.04 * up, p + 0.08 * n),
                             (WIDE, p + 1.1 * n + 0.45 * side + 0.55 * up, p + 0.2 * n)):  # fmt: skip
        c = m.camera_name2id(cam)
        m.cam_pos[c] = eye
        m.cam_quat[c] = T.convert_quat(T.mat2quat(_look_at(eye, target)), to="wxyz")


def run(schedule, cams, size, every, start_phase="free", free_every=None):
    """One expert episode (door_study.run_case's setup); frames every @every steps from @start_phase on"""
    env = make_door_env(arm="hsa", offscreen=True, cylinder_friction=LOCK, **FIXED)
    env.reset()
    aim_cameras(env)
    oracle = DoorOracle(env, wrist_z=SCHEDULES[schedule]["free"][0], perception_error=AIM, max_force=np.inf)
    hook = WristScheduler(schedule, schedules=SCHEDULES, phase_fn=door_phase, order=PHASES)
    rec = dict(t=[], z=[], moment=[], cyl=[], door=[], twist=[], phase=[], frames=[])
    step = {"n": 0}

    def on_step(env, o, row):
        hook(env, o, row)
        phase = door_phase(o, row)
        if PHASES.index(phase) < PHASES.index(start_phase):
            return True
        step["n"] += 1
        rec["t"].append(row["t"])
        rec["z"].append(o.wrist_z)
        rec["moment"].append(np.linalg.norm(row["flange_moment"]))
        rec["cyl"].append(np.degrees(row["cylinder_angle"]))
        rec["door"].append(np.degrees(row["door_angle"]))
        rec["twist"].append(np.degrees(abs(row["wrist_pose"][5])))
        rec["phase"].append(phase)
        k = free_every if (free_every and phase == "free") else every
        if step["n"] % k == 0:
            imgs = {c: render(env, c, *size) for c in cams}
            rec["frames"].append((imgs, len(rec["t"]), phase, row["leg_k"].mean()))
        return True

    result, _ = run_door_episode(env, oracle, max_seconds=40.0, on_step=on_step, reset=False)
    env.close()
    rec["frames"] += [rec["frames"][-1]] * int(round(1.5 * 20 / every))  # hold the last frame
    out = {k: np.array(v) for k, v in rec.items() if k != "frames"}
    out["t"] = out["t"] - out["t"][0]
    out.update(frames=rec["frames"], result=result)
    print(f"{schedule}: {result}, {out['t'][-1]:.1f} s, cylinder {out['cyl'].max():.0f} deg, door {out['door'][-1]:.0f} deg, "
          f"wrist twist {out['twist'].max():.0f} deg")  # fmt: skip
    return out


def phase_bands(d, box, ts, phases, t_max, labels=True):
    """Shade each phase's time span behind a trace and name it at the top"""
    x0, y0, x1, y1 = box
    i = 0
    while i < len(phases):
        j = i
        while j + 1 < len(phases) and phases[j + 1] == phases[i]:
            j += 1
        a, b = x0 + (x1 - x0) * ts[i] / t_max, x0 + (x1 - x0) * ts[j] / t_max
        d.rectangle([a, y0 + 1, max(a + 1, b), y1 - 1], fill=PHASE_FILL[phases[i]])
        if labels and b - a > 26:
            d.text((a + 3, y0 + 2), phases[i], fill=MUTED, font=font(11))
        i = j + 1


def panel(d, box, title, lo, hi, ticks, unit):
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=AXIS)
    for v in ticks:
        y = y1 - (y1 - y0) * (v - lo) / (hi - lo)
        d.line([x0, y, x1, y], fill=AXIS)
        d.text((x0 - 34, y - 7), f"{v:g}{unit}", fill=MUTED, font=font(11))
    d.text((x0, y1 + 4), title, fill=MUTED, font=font(12))


def series(d, box, ts, vs, t_max, lo, hi, color, width=3):
    """trace() with an offset axis: plots (v - lo) on 0..(hi - lo)"""
    trace(d, box, ts, np.clip(np.asarray(vs) - lo, 0, hi - lo), t_max, hi - lo, color)


def save_gif(frames, name, every):
    path = os.path.join(FIGS, name)
    frames = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * every / 20), loop=0, optimize=True)
    print(f"wrote {os.path.relpath(path, os.path.dirname(HERE))} ({len(frames)} frames)")


HERE = os.path.dirname(os.path.abspath(__file__))


def door_demo_gif(every=3, size=(440, 330)):
    sched = "soft_insert_home_turn"
    r = run(sched, (WIDE, CLOSE), size, every, free_every=2 * every)
    W, H = size
    head, plot_h, gap = 84, 120, 20
    ts, t_max = r["t"], r["t"][-1]
    out = []
    for imgs, k, phase, leg_k in r["frames"]:
        canvas = Image.new("RGB", (2 * W + gap, head + H + plot_h + 50), SURFACE)
        d = ImageDraw.Draw(canvas)
        d.text((10, 6), f"door_demos.py episode: schedule {sched}, stiff lock ({LOCK:g} N m), aim error 1 mm / 2 deg",
               fill=INK, font=font(17))  # fmt: skip
        z = r["z"][k - 1]
        done = k == len(ts) and imgs is r["frames"][-1][0]
        status = {"opened": "door open", "cannot turn": "cannot turn"}.get(r["result"], r["result"]) if done else phase
        d.text((10, 34), f"{status}", fill=COLORS[sched], font=font(16))
        d.text((10, 58), f"wrist z {z:+.1f} ({'soft' if z > 0.9 else 'home' if abs(z) < 0.1 else 'stiff' if z < -0.9 else 'switching'}, legs {leg_k:.0f} N/m)   "
               f"cylinder {r['cyl'][k - 1]:4.0f} deg   door {r['door'][k - 1]:4.1f} deg", fill=INK, font=font(15))  # fmt: skip
        canvas.paste(imgs[WIDE], (0, head))
        canvas.paste(imgs[CLOSE], (W + gap, head))
        y0 = head + H + 18
        pw = (2 * W + gap - 3 * 56 - 20) // 3
        boxes = [(46 + i * (pw + 56), y0, 46 + i * (pw + 56) + pw, y0 + plot_h - 20) for i in range(3)]
        for box in boxes:
            phase_bands(d, box, ts, r["phase"], t_max)
        panel(d, boxes[0], "wrist z (+1 soft, -1 stiff)", -1, 1, (-1, 0, 1), "")
        series(d, boxes[0], ts[:k], r["z"][:k], t_max, -1, 1, COLORS[sched])
        panel(d, boxes[1], "moment on the arm's flange", 0, 9, (0, 3, 6, 9), "")
        d.text((boxes[1][2] - 30, boxes[1][3] + 4), "N m", fill=MUTED, font=font(12))
        series(d, boxes[1], ts[:k], r["moment"][:k], t_max, 0, 9, COLORS[sched])
        panel(d, boxes[2], "cylinder angle (blue), door angle x6 (grey), deg", 0, 90, (0, 45, 90), "")
        series(d, boxes[2], ts[:k], r["cyl"][:k], t_max, 0, 90, COLORS[sched])
        series(d, boxes[2], ts[:k], 6 * r["door"][:k], t_max, 0, 90, MUTED)
        out.append(canvas)
    save_gif(out, "door_demo.gif", every)


def door_turn_gif(every=3, size=(440, 330)):
    runs = {s: run(s, (CLOSE,), size, every, start_phase="search") for s in ("soft_insert_home_turn", "fixed_soft")}
    n = max(len(r["frames"]) for r in runs.values())
    t_max = max(r["t"][-1] for r in runs.values())
    W, H = size
    head, plot_h, gap = 84, 120, 20
    out = []
    for i in range(n):
        canvas = Image.new("RGB", (2 * W + gap, head + H + 2 * plot_h + 40), SURFACE)
        d = ImageDraw.Draw(canvas)
        d.text((10, 6), f"Stiff lock ({LOCK:g} N m), same aim error (1 mm, 2 deg), from first contact", fill=INK, font=font(17))
        for c, (s, r) in enumerate(runs.items()):
            imgs, k, phase, leg_k = r["frames"][min(i, len(r["frames"]) - 1)]
            x = c * (W + gap)
            canvas.paste(imgs[CLOSE], (x, head))
            d.text((x + 6, 34), TITLES[s], fill=COLORS[s], font=font(16))
            last = i >= len(r["frames"]) - 1
            status = ({"opened": "door open", "cannot turn": "cannot turn: wrist twists, cylinder stalls"}.get(r["result"], r["result"])
                      if last else phase)  # fmt: skip
            d.text((x + 6, 58), f"{status}   wrist z {r['z'][k - 1]:+.1f}", fill=INK, font=font(15))
            ts = r["t"]
            b1 = (x + 44, head + H + 18, x + W - 10, head + H + plot_h - 2)
            b2 = (x + 44, head + H + plot_h + 22, x + W - 10, head + H + 2 * plot_h + 2)
            for box in (b1, b2):
                phase_bands(d, box, ts, r["phase"], t_max)
            panel(d, b1, "cylinder angle (deg); 90 = unlocked", 0, 90, (0, 45, 90), "")
            series(d, b1, ts[:k], r["cyl"][:k], t_max, 0, 90, COLORS[s])
            panel(d, b2, "wrist twist about the key axis (deg)", 0, 90, (0, 45, 90), "")
            series(d, b2, ts[:k], r["twist"][:k], t_max, 0, 90, COLORS[s])
        out.append(canvas)
    save_gif(out, "door_turn.gif", every)


def main():
    os.makedirs(FIGS, exist_ok=True)
    door_demo_gif()
    door_turn_gif()


if __name__ == "__main__":
    main()
