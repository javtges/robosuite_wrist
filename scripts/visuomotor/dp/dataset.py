"""
Visuomotor door demonstrations (common.py's schema) as Chi-style diffusion-policy training sequences.

    files = find_hdf5(["../data/vm/schedule", "../data/vm/znoise/s0_lock0.4.hdf5"])
    eps = load_episodes(files, keys=obs_keys("wrist_cam,robot"), wrist="z")          # in RAM, images uint8
    train, val = split_episodes(eps, val_ratio=0.05, seed=0)
    norm = fit_normalizer(train, keys)
    ds = SequenceDataset(train, keys, n_obs_steps=2, horizon=16, n_action_steps=8)
    item = ds[0]   # {"obs": {key: (To, ...)}, "action": (Tp, Da)} as torch tensors (images uint8 HWC)

Each sample starts at t (from -(To-1) to L-Tp+(Ta-1)); indices outside the episode repeat its first / last frame.
Observations are the first To frames of the window, actions all Tp. Low-dim observations follow common.policy_obs
(robot0_tcp_rot6d from robot0_tcp_quat, wrist_wrench / WRENCH_SCALE); actions follow common.env_to_policy_action.

--no-cache-images (lazy=True): images stay on disk and are read per sample from h5py (one handle per worker process);
low-dim data is always in RAM.
"""

import glob
import json
import os

import numpy as np
import torch

from visuomotor import common


def find_hdf5(paths):
    """files and directories (searched recursively for *.hdf5) -> sorted unique file list"""
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "**", "*.hdf5"), recursive=True)
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    out = sorted(set(os.path.abspath(f) for f in out))
    assert out, f"no hdf5 files in {paths}"
    return out


def _demo_sort_key(name):
    return int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else name


def _source_keys(keys):
    """policy keys -> the hdf5 obs keys they are computed from"""
    return sorted({"robot0_tcp_quat" if k == "robot0_tcp_rot6d" else k for k in keys})


def file_info(path):
    import h5py

    with h5py.File(path, "r") as f:
        a = f["data"].attrs
        get = lambda k: (a[k].decode() if isinstance(a[k], bytes) else str(a[k])) if k in a else None  # noqa: E731
        return dict(file=path, env_args=get("env_args"), ee_camera=get("ee_camera"), demo_config=get("demo_config"))


def load_episodes(files, keys, wrist, successes_only=False, max_demos=None, seed=0, lazy=False, verbose=True):
    """
    -> list of episodes: dict(obs={key: array}, action=(L, Da) float32, file, demo, success, friction, z0, length).
    @lazy: image keys hold no arrays (read in SequenceDataset). @max_demos: a seeded random subset across all files.
    """
    import h5py

    img_keys = [k for k in keys if k in common.IMAGE_KEYS]
    low_src = [k for k in _source_keys(keys) if k not in common.IMAGE_KEYS]
    index = []
    for path in files:
        with h5py.File(path, "r") as f:
            for name in sorted(f["data"].keys(), key=_demo_sort_key):
                g = f["data"][name]
                success = bool(g.attrs.get("success", True))
                if successes_only and not success:
                    continue
                index.append((path, name, success))
    if max_demos is not None and max_demos < len(index):
        sel = np.sort(np.random.default_rng(seed).choice(len(index), max_demos, replace=False))
        index = [index[i] for i in sel]
    eps = []
    nbytes = 0
    handles = {}
    try:
        for path, name, success in index:
            f = handles.setdefault(path, h5py.File(path, "r"))
            g = f["data"][name]
            raw = {k: g["obs"][k][()] for k in low_src}
            obs = common.policy_obs(raw, [k for k in keys if k not in common.IMAGE_KEYS])
            if not lazy:
                for k in img_keys:
                    obs[k] = np.ascontiguousarray(g["obs"][k][()], dtype=np.uint8)
                    nbytes += obs[k].nbytes
            actions = g["actions"][()]
            eps.append(dict(
                obs=obs,
                action=common.env_to_policy_action(actions, wrist).astype(np.float32),
                wrist_cmd0=np.asarray(g["obs"]["wrist_cmd"][0], dtype=np.float32),
                file=path, demo=name, success=success, length=len(actions),
                friction=float(g.attrs.get("cylinder_friction", np.nan)),
            ))  # fmt: skip
            if verbose and len(eps) % 50 == 0:
                print(f"  loaded {len(eps)}/{len(index)} episodes ({nbytes / 2**30:.1f} GiB images)", flush=True)
    finally:
        for f in handles.values():
            f.close()
    assert eps, "no episodes selected"
    if verbose:
        print(f"loaded {len(eps)} episodes, {sum(e['length'] for e in eps)} steps, {nbytes / 2**30:.2f} GiB images in RAM")
    return eps


def split_episodes(eps, val_ratio=0.05, seed=0):
    """split by episode; at least one validation episode when val_ratio > 0 and there are >= 2 episodes"""
    n = len(eps)
    n_val = int(round(n * val_ratio))
    if val_ratio > 0 and n >= 2:
        n_val = max(1, n_val)
    perm = np.random.default_rng(seed).permutation(n)
    val_idx = set(perm[:n_val].tolist())
    return [e for i, e in enumerate(eps) if i not in val_idx], [e for i, e in enumerate(eps) if i in val_idx]


def fit_normalizer(eps, keys):
    from visuomotor.dp.normalizer import Normalizer

    arrays = {"action": np.concatenate([e["action"] for e in eps])}
    for k in keys:
        if k not in common.IMAGE_KEYS:
            arrays[k] = np.concatenate([e["obs"][k] for e in eps])
    return Normalizer.fit(arrays)


def dataset_stats(eps, keys):
    """per-key min / max / mean, for --dry-run and the run's cfg"""
    out = {"episodes": len(eps), "steps": int(sum(e["length"] for e in eps)), "successes": int(sum(e["success"] for e in eps))}
    act = np.concatenate([e["action"] for e in eps])
    out["action"] = dict(min=act.min(0).round(4).tolist(), max=act.max(0).round(4).tolist())
    for k in keys:
        if k in common.IMAGE_KEYS:
            continue
        x = np.concatenate([e["obs"][k] for e in eps])
        out[k] = dict(min=x.min(0).round(4).tolist(), max=x.max(0).round(4).tolist())
    return out


def initial_wrist_z(eps):
    """median of the first-step wrist_cmd z over the episodes (the policy's starting z for --wrist z)"""
    return float(np.median([e["wrist_cmd0"][2] for e in eps]))


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, eps, keys, n_obs_steps=2, horizon=16, n_action_steps=8, lazy=False):
        self.eps, self.keys = eps, list(keys)
        self.To, self.Tp, self.Ta = n_obs_steps, horizon, n_action_steps
        self.img_keys = [k for k in keys if k in common.IMAGE_KEYS]
        self.lazy = lazy
        self._h5 = {}
        pad_before, pad_after = self.To - 1, self.Ta - 1
        idx = []
        for i, e in enumerate(eps):
            for s in range(-pad_before, e["length"] - self.Tp + pad_after + 1):
                idx.append((i, s))
        self.index = np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def _read_images(self, e, k, t):
        import h5py

        if self._h5.get("pid") != os.getpid():  # one handle set per worker process
            self._h5 = {"pid": os.getpid()}
        f = self._h5.get(e["file"])
        if f is None:
            f = self._h5[e["file"]] = h5py.File(e["file"], "r")
        lo, hi = int(t.min()), int(t.max())
        return f["data"][e["demo"]]["obs"][k][lo:hi + 1][t - lo]

    def __getitem__(self, i):
        ei, s = self.index[i]
        e = self.eps[ei]
        t_all = np.clip(np.arange(s, s + self.Tp), 0, e["length"] - 1)
        t_obs = t_all[:self.To]
        obs = {}
        for k in self.keys:
            if k in self.img_keys and self.lazy:
                obs[k] = torch.from_numpy(np.ascontiguousarray(self._read_images(e, k, t_obs)))
            else:
                obs[k] = torch.from_numpy(np.ascontiguousarray(e["obs"][k][t_obs]))
        return {"obs": obs, "action": torch.from_numpy(e["action"][t_all])}


def provenance(files, eps_train, eps_val):
    """what the run was trained on, for cfg.json"""
    infos = [file_info(f) for f in files]
    count = lambda eps, f: sum(e["file"] == f for e in eps)  # noqa: E731
    return dict(
        files=[dict(file=i["file"], demo_config=json.loads(i["demo_config"]) if i["demo_config"] else None,
                    n_train=count(eps_train, i["file"]), n_val=count(eps_val, i["file"])) for i in infos],
        n_train_episodes=len(eps_train),
        n_val_episodes=len(eps_val),
        n_train_steps=int(sum(e["length"] for e in eps_train)),
    )  # fmt: skip


def check_env_args(files):
    """env_args / ee_camera of the first file; warn when the others disagree beyond cylinder_friction"""
    infos = [file_info(f) for f in files]
    ref = infos[0]

    def strip(env_args):
        if env_args is None:
            return None
        d = json.loads(env_args)
        d.get("env_kwargs", {}).pop("cylinder_friction", None)
        d.pop("env_version", None)
        return d

    for i in infos[1:]:
        if strip(i["env_args"]) != strip(ref["env_args"]):
            print(f"WARNING: env_args of {i['file']} differ from {ref['file']} (beyond cylinder_friction)")
        if i["ee_camera"] != ref["ee_camera"]:
            print(f"WARNING: ee_camera of {i['file']} differs from {ref['file']}")
    return ref["env_args"], ref["ee_camera"]
