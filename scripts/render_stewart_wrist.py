"""
Validation for the PandaStewartAloha embodiment: renders screenshots in standard tabletop tasks, the wrist at
the limits of its workspace along every axis, and runs a scripted pick in Lift.

    python scripts/render_stewart_wrist.py [--out screenshots]
"""
import argparse
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import robosuite as suite

TASKS = ["Lift", "Stack", "PickPlaceCan", "NutAssemblySquare", "Door"]
CAMERAS = ["frontview", "agentview", "robot0_eye_in_hand"]
W, H = 640, 480


def make(task, **kwargs):
    return suite.make(
        task,
        robots="PandaStewartAloha",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        **kwargs,
    )


def render(env, camera):
    return env.sim.render(camera_name=camera, width=W, height=H)[::-1]


def label(img, text):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.rectangle(draw.textbbox((8, 8), text), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return img


def grid(images, cols):
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    out = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(images):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out


def task_screenshots(out):
    tiles = []
    for task in TASKS:
        env = make(task)
        env.reset()
        for cam in CAMERAS:
            img = label(render(env, cam), f"{task} / {cam}")
            img.save(os.path.join(out, f"{task}_{cam}.png"))
            tiles.append(img)
        env.close()
    grid(tiles, len(CAMERAS)).save(os.path.join(out, "all_tasks.png"))


def workspace_limits(out):
    """One row per wrist axis: [-limit, home, +limit], viewed so the motion is in the image plane."""
    env = make("Lift", initialization_noise=None)
    env.reset()
    gripper = env.robots[0].gripper["right"]
    wrist = gripper.wrist
    model, data = env.sim.model._model, env.sim.data._data
    base_id = model.body(gripper.root_body).id
    top_id = model.body(gripper.correct_naming(wrist.mount_body)).id
    axes = data.xmat[base_id].reshape(3, 3).copy()  # wrist frame axes, in world

    tw, th = 560, 420
    renderer = mujoco.Renderer(model, height=th, width=tw)
    cam = mujoco.MjvCamera()
    vis = mujoco.MjvOption()
    vis.geomgroup[0] = 0  # hide collision geoms, as robosuite's own renderer does

    def look_along(d, distance=0.42):
        d = d / np.linalg.norm(d)
        cam.azimuth, cam.elevation, cam.distance = np.rad2deg(np.arctan2(d[1], d[0])), np.rad2deg(np.arcsin(d[2])), distance

    def side_view(axis):
        # horizontal direction perpendicular to @axis (falls back to world x if @axis is vertical)
        d = np.cross(axis, [0, 0, 1])
        return d if np.linalg.norm(d) > 1e-3 else np.array([1.0, 0, 0])

    rows = [
        ("x translation", 0, "mm", -side_view(axes[:, 0])),
        ("y translation", 1, "mm", side_view(axes[:, 1])),
        ("z extension", 2, "mm", side_view(axes[:, 2])),
        ("roll tilt", 3, "deg", axes[:, 0]),
        ("pitch tilt", 4, "deg", axes[:, 1]),
        ("yaw twist", 5, "deg", side_view(axes[:, 2]) + np.array([0, 0, 1.2])),  # from below, looking up
    ]
    tiles = []
    for name, ax, unit, view in rows:
        look_along(view)
        for sign in (-1, 0, 1):
            action = np.zeros(env.action_dim)
            action[6 + ax] = sign
            for _ in range(20):
                obs, *_ = env.step(action)
            cam.lookat[:] = 0.5 * (data.xpos[base_id] + data.xpos[top_id])
            renderer.update_scene(data, camera=cam, scene_option=vis)
            q = obs["robot0_gripper_qpos"][2:8]
            cmd = sign * wrist.action_scale[ax] * (1000 if unit == "mm" else 180 / np.pi)
            meas = q[ax] * (1000 if unit == "mm" else 180 / np.pi)
            tiles.append(label(renderer.render(), f"{name}: commanded {cmd:+.1f} {unit}, measured {meas:+.1f} {unit}"))
    renderer.close()
    grid(tiles, 3).save(os.path.join(out, "wrist_workspace_limits.png"))
    env.close()


def scripted_lift(out):
    """Open-loop pick using the arm only (wrist held at home by a zero wrist action)."""
    env = make("Lift")
    obs = env.reset()

    def step_to(target, grip, n, tol=0.005):
        nonlocal obs
        for _ in range(n):
            action = np.zeros(env.action_dim)
            action[:3] = np.clip((target - obs["robot0_eef_pos"]) / 0.05, -1, 1)
            action[-1] = grip
            obs, *_ = env.step(action)
            if grip == 0 and np.linalg.norm(target - obs["robot0_eef_pos"]) < tol:
                break

    frames = []
    cube = obs["cube_pos"].copy()
    step_to(cube + [0, 0, 0.10], -1, 60)
    frames.append(label(render(env, "frontview"), "above cube"))
    step_to(cube + [0, 0, 0.0], -1, 60)
    frames.append(label(render(env, "frontview"), "descend"))
    step_to(obs["robot0_eef_pos"], 1, 15)
    frames.append(label(render(env, "frontview"), "grasp"))
    step_to(cube + [0, 0, 0.20], 1, 60)
    success = env._check_success()
    frames.append(label(render(env, "frontview"), f"lift: success={success}"))
    grid(frames, 4).save(os.path.join(out, "scripted_lift.png"))
    env.close()
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "screenshots"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    task_screenshots(args.out)
    workspace_limits(args.out)
    print("Scripted Lift success:", scripted_lift(args.out))
    print("Saved screenshots to", os.path.abspath(args.out))
