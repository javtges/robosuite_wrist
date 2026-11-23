"""
CNN diffusion policy (Chi et al., RSS 2023), self-contained: no diffusers, no hydra, no robomimic. Imported under both
venvs (robosuite_env Python 3.14 / torch 2.15, SERL Python 3.10 / torch 2.14): Python 3.10 syntax, no torch.compile.

    obs (B, To, ...) --[one ResNet-18 per camera (GroupNorm, spatial softmax or avg pool) + normalized low-dim]-->
    global condition (B, To * D_obs) --FiLM--> ConditionalUnet1D --epsilon--> DDPM (train) / DDIM (inference)
    -> normalized action chunk (B, Tp, Da)

Images enter as uint8 (B, To, H, W, 3) upright; the model scales them to [0, 1], random-crops (train) or center-crops
(eval) to crop_ratio, and applies ImageNet normalization. Low-dim observations and actions are normalized by the
policy's Normalizer (normalizer.py).

    cfg = default_model_cfg(); policy = DiffusionPolicy(cfg, shapes, normalizer)
    loss = policy.compute_loss(batch); chunk = policy.predict_action(obs)   # (B, Tp, Da), unnormalized
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from visuomotor.dp.normalizer import Normalizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_model_cfg():
    return dict(
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=256,
        cond_predict_scale=True,
        image_pool="spatial_softmax",  # or "avg"
        num_keypoints=32,  # spatial softmax keypoints per camera (feature = 2 * num_keypoints)
        image_feature_dim=64,  # linear projection after pooling
        pretrained_vision=False,
        crop_ratio=0.9,
        num_diffusion_steps=100,
        num_inference_steps=16,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
    )


# ---- vision ------------------------------------------------------------------------------------------------------------
def replace_bn_with_gn(module, features_per_group=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // features_per_group, child.num_features))
        else:
            replace_bn_with_gn(child, features_per_group)
    return module


class SpatialSoftmax(nn.Module):
    """Learned 1x1 conv to K keypoint maps, softmax over space, expected (x, y) per map -> (B, 2K)"""

    def __init__(self, in_channels, num_kp):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_kp, 1)
        self.num_kp = num_kp

    def forward(self, x):
        x = self.conv(x)
        b, k, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        att = F.softmax(x.reshape(b, k, h * w), dim=-1)
        ex = (att * gx.reshape(1, 1, -1)).sum(-1)
        ey = (att * gy.reshape(1, 1, -1)).sum(-1)
        return torch.cat([ex, ey], dim=-1)


def resnet18_trunk(pretrained=False):
    import torchvision

    weights = None
    if pretrained:
        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            net = torchvision.models.resnet18(weights=weights)
        except Exception as e:  # offline: no cached weights
            print(f"[model] ImageNet weights unavailable ({type(e).__name__}: {e}); using random init")
            weights = None
    if weights is None:
        net = torchvision.models.resnet18(weights=None)
    trunk = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc: (B, 512, H/32, W/32)
    return replace_bn_with_gn(trunk)


class ImageEncoder(nn.Module):
    """ResNet-18 trunk + pool + linear for one camera; crops and ImageNet-normalizes uint8 HWC input"""

    def __init__(self, image_hw, cfg):
        super().__init__()
        h, w = image_hw
        self.crop_hw = (int(round(h * cfg["crop_ratio"])), int(round(w * cfg["crop_ratio"])))
        self.trunk = resnet18_trunk(cfg["pretrained_vision"])
        if cfg["image_pool"] == "spatial_softmax":
            self.pool = SpatialSoftmax(512, cfg["num_keypoints"])
            pooled = 2 * cfg["num_keypoints"]
        else:
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            pooled = 512
        self.proj = nn.Linear(pooled, cfg["image_feature_dim"])
        self.out_dim = cfg["image_feature_dim"]
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def crop(self, x):
        """x (N, 3, H, W): per-sample random crop in training, center crop otherwise"""
        n, c, h, w = x.shape
        ch, cw = self.crop_hw
        if (ch, cw) == (h, w):
            return x
        if not self.training:
            i, j = (h - ch) // 2, (w - cw) // 2
            return x[:, :, i:i + ch, j:j + cw]
        iy = torch.randint(0, h - ch + 1, (n,), device=x.device)
        ix = torch.randint(0, w - cw + 1, (n,), device=x.device)
        rows = (iy[:, None] + torch.arange(ch, device=x.device))[:, None, :, None]
        cols = (ix[:, None] + torch.arange(cw, device=x.device))[:, None, None, :]
        return x[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None], rows, cols]

    def forward(self, img):
        """img (N, H, W, 3) uint8 -> (N, out_dim)"""
        x = img.permute(0, 3, 1, 2).float() / 255.0
        x = (self.crop(x) - self.mean) / self.std
        return self.proj(self.pool(self.trunk(x)))


class ObsEncoder(nn.Module):
    """All cameras + normalized low-dim keys at each of To steps -> (B, To * D)"""

    def __init__(self, image_keys, image_hw, lowdim_dims, cfg):
        super().__init__()
        self.image_keys = list(image_keys)
        self.lowdim_keys = list(lowdim_dims)
        self.cams = nn.ModuleDict({k: ImageEncoder(image_hw, cfg) for k in self.image_keys})
        self.step_dim = sum(e.out_dim for e in self.cams.values()) + sum(lowdim_dims.values())

    def forward(self, obs, normalizer):
        feats = []
        b, to = None, None
        for k in self.image_keys:
            x = obs[k]
            b, to = x.shape[:2]
            feats.append(self.cams[k](x.reshape(b * to, *x.shape[2:])).reshape(b, to, -1))
        for k in self.lowdim_keys:
            x = obs[k].float()
            b, to = x.shape[:2]
            feats.append(normalizer.normalize(k, x))
        return torch.cat(feats, dim=-1).reshape(b, -1)


# ---- ConditionalUnet1D (Chi et al.) ----------------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(cin, cout, kernel_size, padding=kernel_size // 2), nn.GroupNorm(n_groups, cout), nn.Mish())

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=True):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(cin, cout, kernel_size, n_groups), Conv1dBlock(cout, cout, kernel_size, n_groups)])
        self.cond_predict_scale = cond_predict_scale
        self.cout = cout
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cout * 2 if cond_predict_scale else cout))
        self.residual_conv = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            scale, bias = embed[:, :self.cout], embed[:, self.cout:]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, down_dims=(256, 512, 1024), kernel_size=5,
                 n_groups=8, cond_predict_scale=True):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed))
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        kw = dict(kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale)
        mid = all_dims[-1]
        self.mid_modules = nn.ModuleList([ConditionalResidualBlock1D(mid, mid, cond_dim, **kw), ConditionalResidualBlock1D(mid, mid, cond_dim, **kw)])
        self.down_modules = nn.ModuleList()
        for i, (cin, cout) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(cin, cout, cond_dim, **kw),
                ConditionalResidualBlock1D(cout, cout, cond_dim, **kw),
                nn.Conv1d(cout, cout, 3, 2, 1) if not last else nn.Identity(),
            ]))  # fmt: skip
        self.up_modules = nn.ModuleList()
        for i, (cin, cout) in enumerate(reversed(in_out[1:])):
            last = i >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(cout * 2, cin, cond_dim, **kw),
                ConditionalResidualBlock1D(cin, cin, cond_dim, **kw),
                nn.ConvTranspose1d(cin, cin, 4, 2, 1) if not last else nn.Identity(),
            ]))  # fmt: skip
        self.final_conv = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups), nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, sample, timestep, global_cond):
        """sample (B, T, Da), timestep (B,) long, global_cond (B, Dc) -> (B, T, Da)"""
        x = sample.transpose(1, 2)
        cond = torch.cat([self.diffusion_step_encoder(timestep), global_cond], dim=-1)
        h = []
        for res1, res2, down in self.down_modules:
            x = res2(res1(x, cond), cond)
            h.append(x)
            x = down(x)
        for m in self.mid_modules:
            x = m(x, cond)
        for res1, res2, up in self.up_modules:
            x = torch.cat([x, h.pop()], dim=1)
            x = res2(res1(x, cond), cond)
            x = up(x)
        return self.final_conv(x).transpose(1, 2)


# ---- noise schedule: DDPM (train) + DDIM (inference) -----------------------------------------------------------------
def make_betas(n, schedule="squaredcos_cap_v2", max_beta=0.999):
    if schedule == "squaredcos_cap_v2":
        f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
        return torch.tensor([min(1 - f((i + 1) / n) / f(i / n), max_beta) for i in range(n)], dtype=torch.float32)
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, n, dtype=torch.float32)
    raise ValueError(f"unknown beta schedule {schedule}")


class NoiseScheduler(nn.Module):
    """Epsilon-prediction DDPM forward process and a deterministic DDIM sampler (eta 0, 'leading' spacing, as diffusers)"""

    def __init__(self, num_train_steps=100, schedule="squaredcos_cap_v2", clip_sample=True):
        super().__init__()
        self.num_train_steps, self.clip_sample = num_train_steps, clip_sample
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - make_betas(num_train_steps, schedule), dim=0), persistent=False)

    def add_noise(self, x0, noise, t):
        ab = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def timesteps(self, n):
        ratio = self.num_train_steps // n
        return [int(i * ratio) for i in reversed(range(n))], ratio

    def ddim_step(self, eps, t, prev_t, x):
        ab = self.alphas_cumprod[t]
        ab_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.ones((), device=x.device)
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        if self.clip_sample:
            x0 = x0.clamp(-1.0, 1.0)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        return ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps


# ---- policy ------------------------------------------------------------------------------------------------------------
class DiffusionPolicy(nn.Module):
    """
    @shapes: dict(image_keys=[...], image_hw=[H, W], lowdim_dims={key: D}, action_dim=Da, n_obs_steps=To, horizon=Tp,
    n_action_steps=Ta). @normalizer: Normalizer with 'action' and every low-dim key.
    """

    def __init__(self, cfg, shapes, normalizer):
        super().__init__()
        self.cfg, self.shapes = cfg, shapes
        self.normalizer = normalizer if normalizer is not None else Normalizer()
        self.obs_encoder = ObsEncoder(shapes["image_keys"], shapes["image_hw"], shapes["lowdim_dims"], cfg)
        self.To, self.Tp, self.Ta, self.Da = shapes["n_obs_steps"], shapes["horizon"], shapes["n_action_steps"], shapes["action_dim"]
        self.unet = ConditionalUnet1D(self.Da, self.obs_encoder.step_dim * self.To, cfg["diffusion_step_embed_dim"], cfg["down_dims"],
                                      cfg["kernel_size"], cfg["n_groups"], cfg["cond_predict_scale"])  # fmt: skip
        self.noise_scheduler = NoiseScheduler(cfg["num_diffusion_steps"], cfg["beta_schedule"], cfg["clip_sample"])
        self.num_inference_steps = cfg["num_inference_steps"]

    def obs_keys(self):
        return self.shapes["image_keys"] + list(self.shapes["lowdim_dims"])

    def compute_loss(self, batch):
        """batch: obs {key: (B, >=To, ...)}, action (B, Tp, Da) unnormalized -> MSE on epsilon"""
        obs = {k: batch["obs"][k][:, :self.To] for k in self.obs_keys()}
        cond = self.obs_encoder(obs, self.normalizer)
        x0 = self.normalizer.normalize("action", batch["action"].float())
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.noise_scheduler.num_train_steps, (x0.shape[0],), device=x0.device)
        pred = self.unet(self.noise_scheduler.add_noise(x0, noise, t), t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def predict_action(self, obs, num_inference_steps=None, generator=None):
        """obs {key: (B, To, ...)} -> unnormalized action chunk (B, Tp, Da)"""
        cond = self.obs_encoder({k: obs[k][:, :self.To] for k in self.obs_keys()}, self.normalizer)
        b = cond.shape[0]
        x = torch.randn((b, self.Tp, self.Da), device=cond.device, generator=generator)
        steps, ratio = self.noise_scheduler.timesteps(num_inference_steps or self.num_inference_steps)
        for t in steps:
            tt = torch.full((b,), t, device=cond.device, dtype=torch.long)
            x = self.noise_scheduler.ddim_step(self.unet(x, tt, cond), t, t - ratio, x)
        return self.normalizer.unnormalize("action", x)


# ---- EMA -------------------------------------------------------------------------------------------------------------
class EMAModel:
    """Chi's power-decay EMA: decay = 1 - (1 + step / inv_gamma) ** -power, clipped to [min_value, max_value]"""

    def __init__(self, model, power=0.75, inv_gamma=1.0, min_value=0.0, max_value=0.9999, update_after_step=0):
        self.averaged_model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.inv_gamma, self.min_value, self.max_value = power, inv_gamma, min_value, max_value
        self.update_after_step = update_after_step
        self.optimization_step = 0
        self.decay = 0.0

    def get_decay(self, step):
        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(1 - (1 + step / self.inv_gamma) ** -self.power, self.max_value))

    @torch.no_grad()
    def step(self, model):
        self.decay = self.get_decay(self.optimization_step)
        for ema_p, p in zip(self.averaged_model.parameters(), model.parameters()):
            if p.requires_grad:
                ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                ema_p.copy_(p.detach())
        for ema_b, b in zip(self.averaged_model.buffers(), model.buffers()):
            ema_b.copy_(b)
        self.optimization_step += 1
