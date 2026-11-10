"""
Images of the KeyInsertion environment and a GIF of the oracle inserting the key:

    cd scripts && python -m key_insertion.render_scene

figures/scene.png       the scene, the key above the lock, and two cutaways (lock walls see-through) of the key going in
                        and seated against the plunger
figures/insertion.gif   stiff vs soft wrist side by side, same 1 mm / 3 deg perception error, cutaway close-up with the
                        push force on the arm's flange traced underneath

The see-through lock is only a rendering change (the lock geoms' alpha), not a different model.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_insertion.evaluate import FIXED_LOCK, PROFILE_ERROR  # noqa: E402
from key_insertion.oracle import KeyOracle, make_env, run_episode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
SURFACE, INK, MUTED, AXIS = (252, 252, 251), (11, 11, 11), (137, 135, 129), (195, 194, 183)
COLORS = {"stiff": (235, 104, 52), "soft": (42, 120, 214)}


def font(size):
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def set_lock_alpha(env, alpha):
    m = env.sim.model._model
    for i in env.lock_geom_ids:
        if m.geom(i).name != "lock_plunger":
            m.geom_rgba[i, 3] = alpha


def render(env, camera, w, h):
    return Image.fromarray(env.sim.render(camera_name=camera, width=w, height=h)[::-1])


def label(img, text, sub=None):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 34 if sub is None else 56], fill=SURFACE)
    d.text((12, 7), text, fill=INK, font=font(19))
    if sub:
        d.text((12, 32), sub, fill=MUTED, font=font(15))
    return img


def scene_png():
    env = make_env(offscreen=True, **FIXED_LOCK)
    oracle = KeyOracle(env, wrist_z=1.0, hold=0.6)
    shots = {}

    def on_step(env, o, row):
        if "hover" not in shots and o.phase == KeyOracle.INSERT:
            set_lock_alpha(env, 1.0)
            shots["hover"] = render(env, "lockview", 800, 600)
        if "mid" not in shots and row["depth"] > 0.012:
            set_lock_alpha(env, 0.25)
            shots["mid"] = render(env, "lockview", 800, 600)
        if "seated" not in shots and o.phase == KeyOracle.HOLD and o.t_phase > 0.25:
            set_lock_alpha(env, 0.25)
            shots["seated"] = render(env, "lockview", 800, 600)

    env.reset()
    set_lock_alpha(env, 1.0)
    shots["front"] = render(env, "frontview", 800, 600)
    run_episode(env, oracle, on_step=on_step, reset=False)
    env.close()
    panels = [
        label(shots["front"], "a. KeyInsertion", "Panda + Stewart wrist + 10 g key; lock randomized on the table"),
        label(shots["hover"], "b. key above the keyhole", "chamfered slot, 0.25 mm clearance per side"),
        label(shots["mid"], "c. going in (lock see-through)", "the red plunger at the bottom pushes back 4 to 10 N"),
        label(shots["seated"], "d. seated", "plunger pushed down 4 mm"),
    ]
    W, H = panels[0].size
    out = Image.new("RGB", (2 * W + 10, 2 * H + 10), SURFACE)
    for i, p in enumerate(panels):
        out.paste(p, ((i % 2) * (W + 10), (i // 2) * (H + 10)))
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, "scene.png")
    out.save(path)
    print(f"wrote {os.path.relpath(path, os.path.dirname(HERE))}")


def trace(d, box, ts, fs, t_max, f_max, color):
    x0, y0, x1, y1 = box
    pts = [(x0 + (x1 - x0) * t / t_max, y1 - (y1 - y0) * min(f, f_max) / f_max) for t, f in zip(ts, fs)]
    if len(pts) > 1:
        d.line(pts, fill=color, width=3)


def insertion_gif(every=3, size=(440, 330)):
    dx, dy, dyaw = PROFILE_ERROR
    runs = {}
    for name, z in (("stiff", -1.0), ("soft", 1.0)):
        env = make_env(offscreen=True, **FIXED_LOCK)
        oracle = KeyOracle(env, wrist_z=z, perception_error=PROFILE_ERROR, hold=0.5)
        frames, forces, times = [], [], []
        step = {"n": 0}

        def on_step(env, o, row, frames=frames, forces=forces, times=times, step=step):
            if o.phase < KeyOracle.ALIGN:
                return
            step["n"] += 1
            times.append(row["t"])
            forces.append(-row["flange_force"][2])
            if step["n"] % every == 0:
                frames.append((render(env, "lockview", *size), len(times), KeyOracle.PHASES[o.phase], row))

        env.reset()
        set_lock_alpha(env, 0.25)
        result, _ = run_episode(env, oracle, on_step=on_step, reset=False)
        # Hold the last frame for a second
        frames += [frames[-1]] * int(round(1.0 * env.control_freq / every))
        runs[name] = (frames, np.array(times) - times[0], np.array(forces), result)
        env.close()

    n = max(len(r[0]) for r in runs.values())
    t_max = max(r[1][-1] for r in runs.values())
    W, H = size
    plot_h, head = 130, 84
    out = []
    for i in range(n):
        canvas = Image.new("RGB", (2 * W + 20, H + head + plot_h + 30), SURFACE)
        d = ImageDraw.Draw(canvas)
        d.text((10, 6), f"Same perception error ({dx * 1e3:g} mm, {np.degrees(dyaw):g} deg), force-blind oracle",
               fill=INK, font=font(17))  # fmt: skip
        for c, name in enumerate(("stiff", "soft")):
            frames, ts, fs, result = runs[name]
            img, k, phase, row = frames[min(i, len(frames) - 1)]
            x = c * (W + 20)
            canvas.paste(img, (x, head))
            title = f"{name} wrist ({row['leg_k'].mean():.0f} N/m legs)"
            d.text((x + 6, 34), title, fill=COLORS[name], font=font(16))
            status = phase if i < len(frames) - 1 else {"seated": "seated", "jammed": "jammed: force stop"}.get(result, result)
            d.text((x + 6, 58), f"{status}   depth {row['depth'] * 1e3:5.1f} mm", fill=INK, font=font(15))
            box = (x + 40, head + H + 20, x + W - 10, head + H + plot_h)
            d.rectangle(box, outline=AXIS)
            d.line([box[0], box[3] - (box[3] - box[1]) * 40 / 45, box[2], box[3] - (box[3] - box[1]) * 40 / 45], fill=AXIS)
            d.text((box[0] - 34, box[1] - 4), "45 N", fill=MUTED, font=font(12))
            d.text((box[0] - 22, box[3] - 12), "0", fill=MUTED, font=font(12))
            d.text((box[0] + 4, box[3] + 4), "push on the arm's flange vs time", fill=MUTED, font=font(12))
            trace(d, box, ts[:k], fs[:k], t_max, 45.0, COLORS[name])
        out.append(canvas)
    path = os.path.join(FIGS, "insertion.gif")
    out = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in out]
    out[0].save(path, save_all=True, append_images=out[1:], duration=int(1000 * every / 20), loop=0, optimize=True)
    print(f"wrote {os.path.relpath(path, os.path.dirname(HERE))} ({len(out)} frames)")


def main():
    scene_png()
    insertion_gif()


if __name__ == "__main__":
    main()
