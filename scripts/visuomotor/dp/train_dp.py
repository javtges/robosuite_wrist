"""
Train a CNN diffusion policy (Chi et al. 2023; model.py) on visuomotor door demonstrations (gen_demos.py datasets).
The policy's inputs are any subset of the observation groups in common.OBS_GROUPS, its targets the tcp pose (position +
6D rotation) plus, for --wrist z, the wrist's normalized z. Logs to wandb; checkpoints hold the EMA weights, the
normalizer and a JSON config, loadable with torch.load(weights_only=True) (policy.py).

Run with robosuite_env from scripts/:
    python -m visuomotor.dp.train_dp --data ../data/vm/schedule --obs wrist_cam,scene_cam,wrench,robot --wrist z \\
        --out ../runs/dp/schedule_all_z --epochs 300 --eval-every 50 --wandb-group schedule
    python -m visuomotor.dp.train_dp --data ../data/vm/static --obs wrench,robot --wrist stiff --out /tmp/x --dry-run
    python -m visuomotor.dp.train_dp --data ../data/vm/static --obs robot --wrist stiff --out /tmp/x --dry-run \\
        --save-init /tmp/x/init.pt     # a randomly initialised checkpoint, to test policy.py / eval_dp.py
    python -m visuomotor.dp.train_dp ... --out ../runs/dp/x --resume   # continue from <out>/train_state.pt

Run dir: cfg.json, latest.pt (every epoch), epoch_<k>.pt (every --checkpoint-every), train_state.pt (raw model, EMA,
optimizer, step, wandb run id: for --resume), eval_*.json and videos from --eval-every.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from visuomotor import common  # noqa: E402
from visuomotor.dp import dataset as D  # noqa: E402
from visuomotor.dp.model import DiffusionPolicy, EMAModel, default_model_cfg  # noqa: E402
from visuomotor.dp.policy import shapes_from_cfg  # noqa: E402


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)


def lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def save_policy(path, ema_model, normalizer, cfg, epoch, step):
    tmp = path + ".tmp"
    torch.save(dict(model=ema_model.state_dict(), normalizer=normalizer.to_dict(), cfg=json.dumps(cfg), epoch=epoch,
                    global_step=step), tmp)  # fmt: skip
    os.replace(tmp, path)


def build_parser():
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="hdf5 files and/or directories (searched recursively)")
    p.add_argument("--obs", default="wrist_cam,scene_cam,wrench,robot", help=f"comma-separated subset of {list(common.OBS_GROUPS)}")
    p.add_argument("--wrist", default="z", choices=list(common.POLICY_WRIST), help="stiff / medium: fixed z; z: the policy's z")
    p.add_argument("--out", required=True, help="run directory")
    # optimisation
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm (0: off)")
    # sequence
    p.add_argument("--n-obs-steps", type=int, default=2, help="To")
    p.add_argument("--horizon", type=int, default=16, help="Tp")
    p.add_argument("--n-action-steps", type=int, default=8, help="Ta")
    # model
    p.add_argument("--num-diffusion-steps", type=int, default=100)
    p.add_argument("--num-inference-steps", type=int, default=16, help="DDIM steps at inference")
    p.add_argument("--crop-ratio", type=float, default=0.9)
    p.add_argument("--down-dims", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--image-pool", default="spatial_softmax", choices=["spatial_softmax", "avg"])
    p.add_argument("--pretrained-vision", action="store_true", help="ImageNet ResNet-18 weights (needs them cached or online)")
    # data
    p.add_argument("--val-ratio", type=float, default=0.05, help="fraction of episodes held out")
    p.add_argument("--successes-only", action="store_true")
    p.add_argument("--max-demos", type=int, default=None, help="random subset of this many episodes (seeded)")
    p.add_argument("--no-cache-images", action="store_true", help="read images from disk per sample instead of RAM")
    p.add_argument("--num-workers", type=int, default=4)
    # bookkeeping
    p.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, cuda:1, mps, cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=50, help="epochs between kept epoch_<k>.pt (latest.pt: every epoch)")
    p.add_argument("--val-every", type=int, default=5, help="epochs between validation losses")
    p.add_argument("--eval-every", type=int, default=0, help="epochs between in-process rollouts (0: off)")
    p.add_argument("--eval-episodes", type=int, default=20, help="per lock friction")
    p.add_argument("--eval-workers", type=int, default=8)
    p.add_argument("--eval-video", type=int, default=2)
    p.add_argument("--eval-max-seconds", type=float, default=40.0)
    p.add_argument("--resume", action="store_true", help="continue from <out>/train_state.pt")
    p.add_argument("--dry-run", action="store_true", help="build data + model, one loss/backward, print, exit (no wandb)")
    p.add_argument("--save-init", default=None, help="with --dry-run: save the randomly initialised policy here")
    # wandb
    p.add_argument("--wandb-project", default=env("WANDB_PROJECT", "hsa-door-visuomotor"))
    p.add_argument("--wandb-entity", default=env("WANDB_ENTITY"))
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-name", default=None, help="default: the run dir's name")
    p.add_argument("--wandb-mode", default=env("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])
    p.add_argument("--tags", nargs="*", default=[])
    return p


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    keys = common.obs_keys(args.obs, wrist=args.wrist)
    image_keys = [k for k in keys if k in common.IMAGE_KEYS]
    lazy = args.no_cache_images

    # ---- data ----
    files = D.find_hdf5(args.data)
    print(f"{len(files)} files:\n  " + "\n  ".join(files))
    env_args, ee_camera = D.check_env_args(files)
    eps = D.load_episodes(files, keys, args.wrist, args.successes_only, args.max_demos, args.seed, lazy=lazy)
    train_eps, val_eps = D.split_episodes(eps, args.val_ratio, args.seed)
    normalizer = D.fit_normalizer(train_eps, keys)
    seq = dict(n_obs_steps=args.n_obs_steps, horizon=args.horizon, n_action_steps=args.n_action_steps, lazy=lazy)
    train_ds = D.SequenceDataset(train_eps, keys, **seq)
    val_ds = D.SequenceDataset(val_eps, keys, **seq) if val_eps else None
    sample = train_ds[0]
    image_hw = list(sample["obs"][image_keys[0]].shape[1:3]) if image_keys else None
    lowdim_dims = {k: int(sample["obs"][k].shape[-1]) for k in keys if k not in image_keys}

    # ---- config ----
    model_cfg = default_model_cfg()
    model_cfg.update(down_dims=args.down_dims, kernel_size=args.kernel_size, image_pool=args.image_pool,
                     pretrained_vision=args.pretrained_vision, crop_ratio=args.crop_ratio,
                     num_diffusion_steps=args.num_diffusion_steps, num_inference_steps=args.num_inference_steps)  # fmt: skip
    env_kwargs = json.loads(env_args)["env_kwargs"] if env_args else {}
    cfg = dict(
        obs_groups=args.obs, obs_keys=keys, image_keys=image_keys, lowdim_dims=lowdim_dims, image_hw=image_hw,
        wrist=args.wrist, action_dim=common.policy_action_dim(args.wrist), n_obs_steps=args.n_obs_steps,
        horizon=args.horizon, n_action_steps=args.n_action_steps, control_freq=float(env_kwargs.get("control_freq", 20.0)),
        z0=D.initial_wrist_z(train_eps), model=model_cfg, env_args=env_args, ee_camera=ee_camera,
        dataset=D.provenance(files, train_eps, val_eps), train=vars(args),
    )  # fmt: skip
    with open(os.path.join(args.out, "cfg.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    policy = DiffusionPolicy(model_cfg, shapes_from_cfg(cfg), normalizer).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    n_vision = sum(p.numel() for p in policy.obs_encoder.parameters())
    print(f"device {device}; obs keys {keys}; wrist {args.wrist}; action dim {cfg['action_dim']}; z0 {cfg['z0']:+.3f}")
    print(f"episodes: {len(train_eps)} train / {len(val_eps)} val; samples {len(train_ds)} train / {len(val_ds) if val_ds else 0} val")
    print(f"params: {n_params / 1e6:.2f} M total, {n_vision / 1e6:.2f} M obs encoder; cond dim {policy.obs_encoder.step_dim} x To={args.n_obs_steps}")

    loader_kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw) if val_ds else None

    if args.dry_run:
        print("dataset stats (train):", json.dumps(D.dataset_stats(train_eps, keys)))
        batch = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=min(args.batch_size, 8), shuffle=True)))
        for k, v in batch["obs"].items():
            print(f"  obs {k:30s} {tuple(v.shape)} {v.dtype}")
        print(f"  action {'':27s} {tuple(batch['action'].shape)} {batch['action'].dtype}")
        policy.train()
        t0 = time.time()
        loss = policy.compute_loss(to_device(batch, device))
        loss.backward()
        print(f"loss {loss.item():.4f} (forward + backward {time.time() - t0:.2f} s)")
        policy.eval()
        chunk = policy.predict_action({k: v.to(device) for k, v in batch["obs"].items()}, num_inference_steps=2)
        print(f"predict_action -> {tuple(chunk.shape)}")
        if args.save_init:
            save_policy(args.save_init, policy, normalizer, cfg, 0, 0)
            print(f"saved randomly initialised checkpoint {args.save_init}")
        return

    # ---- training state ----
    ema = EMAModel(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    start_epoch, step, run_id = 0, 0, None
    state_path = os.path.join(args.out, "train_state.pt")
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(st["model"])
        ema.averaged_model.load_state_dict(st["ema"])
        ema.optimization_step = st["ema_step"]
        opt.load_state_dict(st["optimizer"])
        start_epoch, step, run_id = st["epoch"] + 1, st["global_step"], st.get("wandb_run_id") or None
        print(f"resumed from {state_path}: epoch {start_epoch}, step {step}")

    import wandb

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                     name=args.wandb_name or os.path.basename(os.path.normpath(args.out)), mode=args.wandb_mode,
                     tags=args.tags or None, dir=args.out, id=run_id, resume="allow" if run_id else None,
                     config=dict(train=vars(args), model=model_cfg, obs_keys=keys, n_params=n_params, z0=cfg["z0"],
                                 n_train_episodes=len(train_eps), n_val_episodes=len(val_eps)))  # fmt: skip
    run_id = run.id if args.wandb_mode != "disabled" else ""

    for epoch in range(start_epoch, args.epochs):
        policy.train()
        t0, losses = time.time(), []
        for batch in train_dl:
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_factor(step, args.warmup_steps, total_steps)
            loss = policy.compute_loss(to_device(batch, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            opt.step()
            ema.step(policy)
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                run.log({"train/loss_step": loss.item(), "train/lr": opt.param_groups[0]["lr"], "train/ema_decay": ema.decay}, step=step)
        log = {"train/loss": float(np.mean(losses)), "train/lr": opt.param_groups[0]["lr"], "epoch": epoch,
               "time/epoch_s": time.time() - t0}  # fmt: skip
        if val_dl is not None and (epoch + 1) % args.val_every == 0:
            ema_model = ema.averaged_model.eval()
            vl, mse = [], []
            with torch.no_grad():
                for i, batch in enumerate(val_dl):
                    batch = to_device(batch, device)
                    vl.append(ema_model.compute_loss(batch).item())
                    if i == 0:  # action error of sampled chunks on one batch
                        pred = ema_model.predict_action(batch["obs"])
                        mse.append(torch.mean((pred - batch["action"]) ** 2).item())
            log.update({"val/loss": float(np.mean(vl)), "val/action_mse": float(np.mean(mse))})
        save_policy(os.path.join(args.out, "latest.pt"), ema.averaged_model, normalizer, cfg, epoch, step)
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            save_policy(os.path.join(args.out, f"epoch_{epoch + 1:04d}.pt"), ema.averaged_model, normalizer, cfg, epoch, step)
        torch.save(dict(model=policy.state_dict(), ema=ema.averaged_model.state_dict(), ema_step=ema.optimization_step,
                        optimizer=opt.state_dict(), epoch=epoch, global_step=step, wandb_run_id=run_id), state_path + ".tmp")  # fmt: skip
        os.replace(state_path + ".tmp", state_path)
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in log.items()), flush=True)
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            from visuomotor.dp.eval_dp import evaluate

            # logs its metrics and videos into this run at this step
            evaluate(os.path.join(args.out, "latest.pt"), episodes=args.eval_episodes, workers=args.eval_workers,
                     seed=args.seed + 1000, max_seconds=args.eval_max_seconds, video=args.eval_video,
                     out=os.path.join(args.out, f"eval_epoch_{epoch + 1:04d}.json"), wandb_run=run, step=step)  # fmt: skip
        run.log(log, step=step)
    run.finish()


if __name__ == "__main__":
    main()
