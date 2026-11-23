"""
Roll out a trained diffusion policy (train_dp.py checkpoint) in KeyDoor, rebuilt from the checkpoint's env_args and
ee_camera, per lock friction. Each episode: env.reset(), DoorOracle.reset() only to settle the wrist for 0.5 s at the
policy's initial wrist z (as gen_demos.py does), then the policy acts every control step until the door opens
(env._check_success()), the flange force passes --max-force, or --max-seconds of sim time.

Run with robosuite_env from scripts/ (on a headless Linux server MUJOCO_GL defaults to egl):
    python -m visuomotor.dp.eval_dp --ckpt ../runs/dp/x/latest.pt --episodes 50 --workers 16 --video 4
    python -m visuomotor.dp.eval_dp --ckpt ../runs/dp/x/latest.pt --episodes 2 --workers 1 --max-seconds 3
    python -m visuomotor.dp.eval_dp --ckpt ../runs/dp/x/latest.pt --wandb-run-id <id>   # log into the training run

--episodes is per lock friction. Writes --out (default <ckpt dir>/eval_<ckpt stem>.json): per-episode rows and a
summary overall and per friction (success rate, time to success, peak flange force, reached seated / unlocked /
opened, max cylinder and door angles, mean wrist z). Videos (wrist | scene camera) go next to it. Each row also has the
key tip's aim in the keyhole frame (tip_aim) at first contact (flange force > CONTACT_N), on reaching the lock face (entry, depth >= ENTRY_MM), at its closest approach to
the keyhole and at the end.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_W = {}  # per-process worker state: runner, envs by friction, settings


def _init_worker(ckpt, num_inference_steps, settings, threads):
    import torch

    from visuomotor.dp.policy import DiffusionPolicyRunner

    if threads:
        torch.set_num_threads(threads)
    _W.update(runner=DiffusionPolicyRunner(ckpt, "cpu", num_inference_steps), envs={}, settings=settings)


def _env(friction, cameras):
    from visuomotor import common

    key = (friction, cameras)
    if key not in _W["envs"]:
        cfg = _W["runner"].cfg
        _W["envs"][key] = common.make_env(cfg["env_args"], cfg["ee_camera"], cameras=cameras, cylinder_friction=friction)
    return _W["envs"][key]


def _write_video(frames, path, fps):
    try:
        import cv2

        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()
        return path
    except ImportError:
        from PIL import Image

        path = path[:-4] + ".gif"
        ims = [Image.fromarray(f) for f in frames[::2]]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(2000 / fps), loop=0)
        return path


CONTACT_N = 2.0  # flange force (N) that counts as first contact
ENTRY_MM = -1.0  # tip depth (mm) that counts as reaching the lock face (entry)


def tip_aim(env):
    """The key tip in the keyhole frame, as key_insertion/oracle.py logs it: x across the slot, y along its long side,
    depth below the lock face (mm, negative = still in front), yaw about the keyhole axis and tilt off it (deg)"""
    p, R = env.key_tip_in_lock()
    return dict(x_mm=float(p[0] * 1e3), y_mm=float(p[1] * 1e3), depth_mm=float(-p[2] * 1e3),
                lateral_mm=float(np.linalg.norm(p[:2]) * 1e3), dist_mm=float(np.linalg.norm(p) * 1e3),
                yaw_deg=float(np.degrees(np.arctan2(R[0, 1], R[1, 1]))),
                tilt_deg=float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0)))))  # fmt: skip


def run_episode(task):
    """one rollout in this process's env -> metrics dict (task: friction, index, seed, video path or None)"""
    from key_insertion.door_oracle import DoorOracle
    from visuomotor import common

    runner, s = _W["runner"], _W["settings"]
    video = task.get("video")
    env = _env(task["friction"], runner.needs_cameras or bool(video))
    env.rng = np.random.default_rng(task["seed"])
    env.reset()
    runner.reset()
    oracle = DoorOracle(env, wrist_z=float(runner.wrist_cmd[2]), max_force=np.inf)
    oracle.reset()  # settles the wrist for 0.5 s
    obs = common.obs_extras(env, env._get_observations(force_update=True), runner.wrist_cmd)
    t0 = env.sim.data.time
    peak_f, cyl_max, door_max, zs, frames = 0.0, -np.inf, -np.inf, [], []
    seated = unlocked = success = False
    contact, entry, closest = None, None, None  # aim at first contact and at the lock face; closest approach
    result, wall = "timeout", time.time()
    while env.sim.data.time < s["max_seconds"]:
        if video:
            frames.append(np.concatenate([obs[k] for k in common.IMAGE_KEYS if k in obs], axis=1))
        action = runner.act(obs)
        raw, _, _, _ = env.step(action)
        obs = common.obs_extras(env, raw, action[6:12])
        f = float(np.linalg.norm(oracle.flange_force()))
        peak_f = max(peak_f, f)
        aim = tip_aim(env)
        if closest is None or aim["dist_mm"] < closest["dist_mm"]:
            closest = aim
        if entry is None and aim["depth_mm"] >= ENTRY_MM:
            entry = dict(aim, t=float(env.sim.data.time - t0), wrist_z=float(runner.wrist_cmd[2]))
        if contact is None and f > CONTACT_N:
            contact = dict(aim, t=float(env.sim.data.time - t0), wrist_z=float(runner.wrist_cmd[2]))
        cyl_max, door_max = max(cyl_max, env.cylinder_angle()), max(door_max, env.door_angle())
        seated |= bool(env.key_seated())
        unlocked |= bool(env.unlocked())
        zs.append(float(runner.wrist_cmd[2]))
        if env._check_success():
            success, result = True, "opened"
            break
        if f > s["max_force"]:
            result = "force_abort"
            break
    t = float(env.sim.data.time - t0)
    out = dict(friction=task["friction"], index=task["index"], seed=task["seed"], success=success, result=result,
               time=t, time_to_success=t if success else None, steps=len(zs), peak_flange_force=peak_f,
               reached_seated=seated, reached_unlocked=unlocked, max_cylinder_deg=float(np.degrees(cyl_max)),
               max_door_deg=float(np.degrees(door_max)), final_door_deg=float(np.degrees(env.door_angle())),
               mean_wrist_z=float(np.mean(zs)) if zs else None, wall_s=time.time() - wall,
               first_contact=contact, entry=entry, closest_approach=closest, final_tip=tip_aim(env))  # fmt: skip
    if video and frames:
        out["video"] = _write_video(frames, video, runner.control_freq)
    return out


def summarize(rows):
    def stats(rs):
        if not rs:
            return {}
        ok = [r for r in rs if r["success"]]
        return dict(
            n=len(rs), success_rate=len(ok) / len(rs),
            time_to_success=float(np.mean([r["time_to_success"] for r in ok])) if ok else None,
            peak_flange_force_mean=float(np.mean([r["peak_flange_force"] for r in rs])),
            peak_flange_force_max=float(np.max([r["peak_flange_force"] for r in rs])),
            seated_rate=float(np.mean([r["reached_seated"] for r in rs])),
            unlocked_rate=float(np.mean([r["reached_unlocked"] for r in rs])),
            force_abort_rate=float(np.mean([r["result"] == "force_abort" for r in rs])),
            max_cylinder_deg_mean=float(np.mean([r["max_cylinder_deg"] for r in rs])),
            max_door_deg_mean=float(np.mean([r["max_door_deg"] for r in rs])),
            mean_wrist_z=float(np.mean([r["mean_wrist_z"] for r in rs if r["mean_wrist_z"] is not None])),
        )  # fmt: skip

    out = {"all": stats(rows)}
    for fr in sorted({r["friction"] for r in rows}):
        out[f"lock{fr:g}"] = stats([r for r in rows if r["friction"] == fr])
    return out


def evaluate(ckpt, episodes=20, workers=1, cylinder_frictions=(0.15, 0.4), seed=0, max_seconds=40.0,
             num_inference_steps=None, video=0, max_force=80.0, out=None, wandb_run=None, step=None):  # fmt: skip
    """-> dict(summary, episodes, wandb={flat metrics}); logs metrics and videos to @wandb_run (at @step) if given"""
    out = out or os.path.join(os.path.dirname(os.path.abspath(ckpt)), f"eval_{os.path.splitext(os.path.basename(ckpt))[0]}.json")
    vid_dir = os.path.splitext(out)[0] + "_videos"
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(episodes):
        for fr in cylinder_frictions:
            tasks.append(dict(friction=float(fr), index=i, seed=int(rng.integers(1 << 31))))
    for t in tasks[:video]:
        os.makedirs(vid_dir, exist_ok=True)
        t["video"] = os.path.join(vid_dir, f"lock{t['friction']:g}_ep{t['index']:03d}.mp4")
    settings = dict(max_seconds=max_seconds, max_force=max_force)
    t0 = time.time()
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        _init_worker(ckpt, num_inference_steps, settings, 0)
        rows = []
        for t in tasks:
            rows.append(run_episode(t))
            _print_row(rows[-1])
    else:
        threads = max(1, (os.cpu_count() or 1) // workers)
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(ckpt, num_inference_steps, settings, threads)) as pool:
            rows = []
            for r in pool.imap_unordered(run_episode, tasks):
                rows.append(r)
                _print_row(r)
    rows.sort(key=lambda r: (r["friction"], r["index"]))
    summary = summarize(rows)
    flat = {}
    for grp, st in summary.items():
        for k, v in st.items():
            if v is not None:
                flat[f"eval/{k}" if grp == "all" else f"eval_{grp}/{k}"] = v
    res = dict(ckpt=os.path.abspath(ckpt), summary=summary, episodes=rows, wandb=flat, wall_s=time.time() - t0,
               settings=dict(settings, episodes=episodes, cylinder_frictions=list(cylinder_frictions), seed=seed,
                             num_inference_steps=num_inference_steps))  # fmt: skip
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"eval: {json.dumps(summary)}\n-> {out} ({res['wall_s']:.0f} s)")
    if wandb_run is not None:
        import wandb

        log = dict(flat)
        for r in rows:
            if r.get("video"):
                log[f"eval_video/lock{r['friction']:g}_ep{r['index']:03d}"] = wandb.Video(r["video"], format=r["video"][-3:])
        wandb_run.log(log, step=step) if step is not None else wandb_run.log(log)
    return res


def _print_row(r):
    print(f"  lock {r['friction']:g} ep {r['index']:3d}: {r['result']:11s} {r['time']:5.1f} s, peak flange {r['peak_flange_force']:5.1f} N, "
          f"seated {int(r['reached_seated'])} unlocked {int(r['reached_unlocked'])}, door {r['max_door_deg']:5.1f} deg, "
          f"wrist z {r['mean_wrist_z'] if r['mean_wrist_z'] is not None else float('nan'):+.2f} ({r['wall_s']:.0f} s wall)", flush=True)  # fmt: skip


def main():
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=20, help="per lock friction")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cylinder-frictions", type=float, nargs="+", default=[0.15, 0.4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=40.0, help="sim time limit per episode (incl. the 0.5 s settle)")
    p.add_argument("--max-force", type=float, default=80.0, help="abort when the flange force exceeds this (N)")
    p.add_argument("--num-inference-steps", type=int, default=None, help="DDIM steps (default: the checkpoint's)")
    p.add_argument("--video", type=int, default=0, help="save (and log) videos of the first N episodes")
    p.add_argument("--out", default=None, help="results json (default <ckpt dir>/eval_<ckpt stem>.json)")
    p.add_argument("--wandb-project", default=env("WANDB_PROJECT", "hsa-door-visuomotor"))
    p.add_argument("--wandb-entity", default=env("WANDB_ENTITY"))
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-run-id", default=None, help="log into this existing run (resume='allow')")
    p.add_argument("--wandb-mode", default=env("WANDB_MODE"), choices=["online", "offline", "disabled"],
                   help="default: $WANDB_MODE, else online with --wandb-run-id / --wandb-name, else disabled")
    p.add_argument("--tags", nargs="*", default=[])
    args = p.parse_args()

    mode = args.wandb_mode or ("online" if (args.wandb_run_id or args.wandb_name) else "disabled")
    run = None
    if mode != "disabled":
        import wandb

        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, name=args.wandb_name,
                         id=args.wandb_run_id, resume="allow" if args.wandb_run_id else None, mode=mode,
                         tags=args.tags or None, job_type="eval", config=dict(eval=vars(args)))  # fmt: skip
    evaluate(args.ckpt, args.episodes, args.workers, args.cylinder_frictions, args.seed, args.max_seconds,
             args.num_inference_steps, args.video, args.max_force, args.out, wandb_run=run)  # fmt: skip
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
