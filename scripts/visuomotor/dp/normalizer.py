"""
Min-max normalizer to [-1, 1] for the diffusion policy's actions and low-dim observations (Chi et al. 2023's "limits"
mode). Fitted on the training split by dataset.py, stored in the checkpoint as plain tensors (<key>.scale,
<key>.offset), applied on the device by model.py. Imported under both venvs: Python 3.10-compatible, torch + numpy.

    norm = Normalizer.fit({"action": actions, "robot0_tcp_pos": tcp_pos})   # (N, D) arrays
    x = norm.normalize("action", a); a = norm.unnormalize("action", x)
"""

import numpy as np
import torch
from torch import nn


class Normalizer(nn.Module):
    """Per-key affine map x_n = x * scale + offset, with min -> -1 and max -> +1; constant dims map to 0"""

    def __init__(self, params=None):
        super().__init__()
        self.keys = []
        for k, (scale, offset) in (params or {}).items():
            self.add(k, scale, offset)

    def add(self, key, scale, offset):
        name = key.replace(".", "_")
        self.register_buffer(name + "__scale", torch.as_tensor(np.asarray(scale), dtype=torch.float32))
        self.register_buffer(name + "__offset", torch.as_tensor(np.asarray(offset), dtype=torch.float32))
        self.keys.append(key)

    @classmethod
    def fit(cls, arrays, eps=1e-4):
        params = {}
        for k, x in arrays.items():
            x = np.asarray(x, dtype=np.float64).reshape(-1, np.shape(x)[-1])
            lo, hi = x.min(0), x.max(0)
            rng = hi - lo
            const = rng < eps
            scale = np.where(const, 1.0, 2.0 / np.where(const, 1.0, rng))
            offset = np.where(const, -lo, -1.0 - scale * lo)  # constant dims: x - lo = 0
            params[k] = (scale, offset)
        return cls(params)

    def _p(self, key):
        name = key.replace(".", "_")
        return getattr(self, name + "__scale"), getattr(self, name + "__offset")

    def normalize(self, key, x):
        s, o = self._p(key)
        return x * s + o

    def unnormalize(self, key, x):
        s, o = self._p(key)
        return (x - o) / s

    # ---- checkpoint io: {key.scale: tensor, key.offset: tensor} -------------------------------------------------------
    def to_dict(self):
        out = {}
        for k in self.keys:
            s, o = self._p(k)
            out[k + ".scale"], out[k + ".offset"] = s.detach().cpu().clone(), o.detach().cpu().clone()
        return out

    @classmethod
    def from_dict(cls, d):
        keys = sorted({k.rsplit(".", 1)[0] for k in d})
        return cls({k: (d[k + ".scale"], d[k + ".offset"]) for k in keys})
