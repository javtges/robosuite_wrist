"""
Run a trained diffusion policy (train_dp.py checkpoint) in the door env, one robosuite action per control step.
Imported under both venvs (robosuite_env and the SERL venv): Python 3.10 syntax, torch + numpy + scipy only.

    from visuomotor.dp.policy import DiffusionPolicyRunner
    runner = DiffusionPolicyRunner("runs/dp/x/latest.pt", device="cpu", num_inference_steps=16)
    runner.reset()
    obs = obs_extras(env, env._get_observations(force_update=True), runner.wrist_cmd)   # one step, upright images
    while ...:
        action = runner.act(obs)                # (12,) robosuite action
        raw, *_ = env.step(action)
        obs = obs_extras(env, raw, action[6:12])
    runner.base_action   # the policy action behind the last act(), unnormalized (9 or 10: tcp pos, rot6d[, wrist z])
    runner.wrist_cmd     # the wrist part of the last act() (6), or the initial command after reset()

The runner keeps the last To observations (padded with the first after reset()), predicts a Tp-step chunk when its
queue is empty and executes its steps To-1 .. To-1+Ta-1 (Chi et al.). The wrist part comes from common.WristCommand:
a fixed z for --wrist stiff / medium, the policy's own z rate-limited at 1 z/s for --wrist z (starting at cfg z0).
"""

import json
from collections import deque

import numpy as np
import torch

from visuomotor import common
from visuomotor.dp.model import DiffusionPolicy
from visuomotor.dp.normalizer import Normalizer


def shapes_from_cfg(cfg):
    return dict(image_keys=list(cfg["image_keys"]), image_hw=cfg["image_hw"], lowdim_dims=dict(cfg["lowdim_dims"]),
                action_dim=cfg["action_dim"], n_obs_steps=cfg["n_obs_steps"], horizon=cfg["horizon"],
                n_action_steps=cfg["n_action_steps"])  # fmt: skip


def load_checkpoint(path, device="cpu"):
    """train_dp.py checkpoint -> (DiffusionPolicy with the EMA weights, eval mode, cfg dict)"""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = json.loads(ckpt["cfg"])
    policy = DiffusionPolicy(cfg["model"], shapes_from_cfg(cfg), Normalizer.from_dict(ckpt["normalizer"]))
    policy.load_state_dict(ckpt["model"])
    return policy.to(device).eval(), cfg


class DiffusionPolicyRunner:
    def __init__(self, ckpt_path, device="cpu", num_inference_steps=None):
        self.device = torch.device(device)
        self.policy, self.cfg = load_checkpoint(ckpt_path, self.device)
        self.num_inference_steps = num_inference_steps or self.cfg["model"]["num_inference_steps"]
        self.obs_keys = list(self.cfg["obs_keys"])
        self.wrist = self.cfg["wrist"]
        self.needs_cameras = any(k in common.IMAGE_KEYS for k in self.obs_keys)
        self.n_obs_steps = int(self.cfg["n_obs_steps"])
        self.n_action_steps = int(self.cfg["n_action_steps"])
        self.horizon = int(self.cfg["horizon"])
        self.control_freq = float(self.cfg.get("control_freq", 20.0))
        self.z0 = self.cfg.get("z0")
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.n_obs_steps)
        self.queue = deque()
        self.wrist_command = common.WristCommand(self.wrist, control_freq=self.control_freq, z0=self.z0)
        self.wrist_cmd = self.wrist_command()
        self.base_action = None
        self.last_chunk = None

    def _batch(self, obs_history):
        """list of To policy_obs dicts -> {key: (1, To, ...)} tensors on the device"""
        out = {}
        for k in self.obs_keys:
            x = np.stack([o[k] for o in obs_history])
            out[k] = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return out

    def predict_chunk(self, obs_history):
        """To raw observation dicts (obs_extras format, oldest first) -> unnormalized (Tp, Da) chunk"""
        assert len(obs_history) == self.n_obs_steps, f"need {self.n_obs_steps} observations, got {len(obs_history)}"
        hist = [common.policy_obs(o, self.obs_keys) for o in obs_history]
        chunk = self.policy.predict_action(self._batch(hist), self.num_inference_steps)
        return chunk[0].cpu().numpy().astype(np.float64)

    def act(self, obs):
        """one step's observation (obs_extras format) -> (12,) robosuite action"""
        self.history.append(obs)
        while len(self.history) < self.n_obs_steps:
            self.history.appendleft(obs)
        if not self.queue:
            chunk = self.predict_chunk(list(self.history))
            self.last_chunk = chunk
            start = self.n_obs_steps - 1
            self.queue.extend(chunk[start:start + self.n_action_steps])
        self.base_action = self.queue.popleft()
        z_target = float(self.base_action[9]) if len(self.base_action) > 9 else None
        self.wrist_cmd = self.wrist_command(z_target)
        return common.policy_to_env_action(self.base_action, self.wrist_cmd)
