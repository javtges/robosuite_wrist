"""
Wrist-side exploration for residual fine-tuning: a perturbation added to the WRIST part of the executed action only
(executed = base + agent residual + perturbation, then clipped to the stroke). Units are the wrist's normalized pose
command (x, y, z, rx, ry, rz in [-1, 1]; physical = normalized * wrist.action_scale, i.e. xy_range, z_range,
tilt_range, yaw_range, as StewartPlatformWrist.format_action maps it). Every sampler acts on the active wrist axes of the
env's structure (wrist_z: z only).

    from visuomotor.residual.exploration import make_sampler
    sampler = make_sampler(FLAGS, active_axes=(2,))      # FLAGS or any object with the attributes in DEFAULTS
    env.unwrapped.set_sampler(sampler)                   # door_residual_env calls reset() / sample() itself

    python -m visuomotor.residual.exploration            # self-checks (no env except the force-budget one)

--wrist_noise:
    none        no perturbation
    random      i.i.d. each step on the active axes: gaussian with std --noise_std, or uniform on +-noise_std
                (--noise_dist uniform)
    gaussian    structured N(0, Sigma) restricted to the active axes. Sigma (6x6, normalized units) from
                  --noise_cov_source flags: per-axis stds --noise_std (1 or 6 values), with an optional correlation
                    matrix --noise_corr (.npy, 6x6), or a full covariance --noise_cov (.npy, 6x6)
                  --noise_cov_source force_budget: the CLAUDE.md water-filling ("Exploration noise under a force
                    budget"): max log det Sigma s.t. tr(R Sigma) <= --noise_budget, Sigma_ii <= cap_i^2, with
                    R = Kn^T W_f Kn, Kn = K diag(action_scale) (wrench per normalized unit), K the wrist's
                    platform_stiffness at the nominal commanded pose (recomputed every --noise_cov_every steps),
                    W_f = diag(1/f_max^2 x3, 1/tau_max^2 x3) (--noise_f_max N, --noise_tau_max N m), caps
                    --noise_caps (normalized std, default 1 = the stroke). K is the wrist alone, i.e. the series
                    stiffness against a rigid environment in every direction (the conservative K_s).
                --noise_tau > 0: Ornstein-Uhlenbeck with time constant tau (s) and the same stationary covariance
                (x <- a x + sqrt(1 - a^2) L eps, a = exp(-dt / tau), L L^T = Sigma); 0 = white
    ergodic     spectral multiscale coverage (Mathew & Mezic 2011) over --ergodic_axes (subset of the active axes,
                default all of them) in the normalized box [-1, 1]^d: u = -u_max B / |B|,
                B = sum_k Lambda_k S_k grad F_k(x), F_k a cosine basis with --ergodic_modes per dim, S_k = t (c_k -
                phi_k) from the episode's time-averaged coefficients c_k of the executed wrist command. The target
                density phi is estimated online from the leg-sensed wrench at visited poses (TargetDensity; kept across
                episodes): low_force (default) phi ~ exp(-f_hat(x)^2 / f_scale^2), f_hat a kernel regression of |force|
                shrunk toward 0 where nothing was visited (unvisited poses look attractive); force_variance phi ~
                local variance of |force| (prior f_scale^2 where unvisited); uniform. u is the perturbation itself
                (an offset of size u_max from the agent's command, not integrated).
                FIRST VERSION, meant to be reworked: the offset form, the force-only target and the evaluation set
                (a 32^d grid for d <= 2, else 4096 fixed uniform points) are all placeholders.
"""

import warnings

import numpy as np

AXES = ("x", "y", "z", "rx", "ry", "rz")
WRIST_NOISE = ("none", "random", "gaussian", "ergodic")
DEFAULTS = dict(
    wrist_noise="none",
    noise_std=[0.1],
    noise_dist="gaussian",
    noise_corr=None,
    noise_cov=None,
    noise_cov_source="flags",
    noise_f_max=5.0,
    noise_tau_max=0.25,
    noise_budget=1.0,
    noise_caps=[1.0],
    noise_cov_every=10,
    noise_tau=0.0,
    ergodic_axes=None,
    ergodic_modes=5,
    ergodic_umax=0.1,
    ergodic_target="low_force",
    ergodic_f_scale=5.0,
    ergodic_bandwidth=0.15,
    ergodic_memory=2000,
    ergodic_update=20,
)


def _vec6(v, name):
    v = np.atleast_1d(np.asarray([float(x) for x in np.atleast_1d(v)]))
    assert len(v) in (1, 6), f"{name}: 1 or 6 values, got {len(v)}"
    return np.broadcast_to(v, (6,)).copy()


# ---- water-filling (CLAUDE.md, "Exploration noise under a force budget") ---------------------------------------------
def water_fill(R, caps, budget):
    """
    max log det Sigma s.t. tr(R Sigma) <= budget, Sigma_ii <= caps_i (variances). Solved in the eigenbasis of R (noise
    axes = eigenvectors of R) by reverse water-filling, sigma_i^2 = min(cap_i, 1 / (lambda r_i)). Caps are per pose
    axis; in the eigenbasis they are approximated by cap_i = 1 / sum_j V_ji^2 / caps_j (exact for a diagonal R), and
    afterwards Sigma is rescaled so that both the caps and the budget hold exactly.
    """
    R = 0.5 * (R + R.T)
    r, V = np.linalg.eigh(R)
    r = np.clip(r, 0.0, None)
    c = 1.0 / ((V**2) / caps[:, None]).sum(axis=0)
    var = lambda lam: np.minimum(c, 1.0 / np.maximum(lam * r, 1e-300))  # noqa: E731
    if r @ c <= budget:
        s2 = c
    else:
        lo, hi = -30.0, 30.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if r @ var(10**mid) > budget else (lo, mid)
        s2 = var(10**hi)
    S = V @ np.diag(s2) @ V.T
    d = np.minimum(1.0, np.sqrt(caps / np.maximum(np.diag(S), 1e-300)))
    S = d[:, None] * S * d[None, :]
    load = np.trace(R @ S)
    if load > budget:
        S *= budget / load
    return 0.5 * (S + S.T)


def force_budget_cov(K, action_scale, f_max, tau_max, budget, caps, axes):
    """Sigma (len(axes) x len(axes), normalized units) for the stiffness @K (6x6, pose coordinates) on @axes"""
    Kn = K @ np.diag(action_scale)
    Wf = np.diag([1.0 / f_max**2] * 3 + [1.0 / tau_max**2] * 3)
    R = Kn.T @ Wf @ Kn
    ax = np.asarray(axes)
    return water_fill(R[np.ix_(ax, ax)], np.asarray(caps, dtype=float)[ax] ** 2, budget)


# ---- samplers --------------------------------------------------------------------------------------------------------
class WristSampler:
    """Interface: reset(env, rng) at each episode start; sample(env, obs_info, dt) -> perturbation on self.axes"""

    def __init__(self, axes):
        self.axes = np.asarray(axes, dtype=int)
        self.d = len(self.axes)
        self.rng = np.random.default_rng()

    def reset(self, env, rng):
        self.rng = rng

    def sample(self, env, obs_info, dt):
        return np.zeros(self.d)


class RandomSampler(WristSampler):
    def __init__(self, axes, std, dist="gaussian"):
        super().__init__(axes)
        self.std = _vec6(std, "noise_std")[self.axes]
        assert dist in ("gaussian", "uniform"), dist
        self.dist = dist

    def sample(self, env, obs_info, dt):
        if self.dist == "uniform":
            return self.rng.uniform(-self.std, self.std)
        return self.rng.normal(0.0, self.std)


class GaussianSampler(WristSampler):
    """
    N(0, Sigma) on the active axes, white or OU (@tau s). @cov: fixed 6x6 Sigma (normalized units), or None with
    @force_budget = dict(f_max, tau_max, budget, caps, every) to recompute Sigma from the wrist's stiffness
    """

    def __init__(self, axes, cov=None, tau=0.0, force_budget=None):
        super().__init__(axes)
        assert (cov is None) != (force_budget is None), "pass cov or force_budget"
        self.fb = force_budget
        self.tau = float(tau)
        self.Sigma = None if cov is None else np.asarray(cov, dtype=float)[np.ix_(self.axes, self.axes)]
        self.x = np.zeros(self.d)
        self.n = 0

    def _update_cov(self, env, obs_info):
        w = env.wrist
        pose = np.asarray(obs_info["wrist_cmd_agent"]) * w.action_scale  # nominal command (not the perturbed one)
        K = w.platform_stiffness(pose)
        fb = self.fb
        self.Sigma = force_budget_cov(K, w.action_scale, fb["f_max"], fb["tau_max"], fb["budget"], fb["caps"], self.axes)

    @property
    def L(self):
        return np.linalg.cholesky(self.Sigma + 1e-12 * np.eye(self.d))

    def reset(self, env, rng):
        super().reset(env, rng)
        self.n = 0
        self.x = None  # drawn from the stationary distribution on the first sample (Sigma may depend on the pose)

    def sample(self, env, obs_info, dt):
        if self.fb is not None and self.n % max(1, int(self.fb["every"])) == 0:
            self._update_cov(env, obs_info)
        self.n += 1
        eps = self.L @ self.rng.standard_normal(self.d)
        if self.tau <= 0 or self.x is None:
            self.x = eps
        else:
            a = np.exp(-dt / self.tau)
            self.x = a * self.x + np.sqrt(1.0 - a * a) * eps
        return self.x.copy()


# ---- ergodic ---------------------------------------------------------------------------------------------------------
class CosineBasis:
    """Normalized cosine basis on [-1, 1]^d (y = x + 1 in [0, 2]^d), modes 0..m-1 per dim (Mathew & Mezic 2011)"""

    def __init__(self, d, modes):
        self.d, self.L = d, 2.0
        self.k = np.array(np.meshgrid(*[np.arange(modes)] * d, indexing="ij")).reshape(d, -1).T  # (K, d)
        self.K = len(self.k)
        self.w = self.k * np.pi / self.L  # (K, d)
        self.h = np.sqrt(np.prod(np.where(self.k > 0, self.L / 2, self.L), axis=1))  # ||cos||_2 per mode
        self.Lambda = (1.0 + (self.k**2).sum(1)) ** (-(d + 1) / 2.0)
        self.vol = self.L**d

    def F(self, x):
        """(N, d) points -> (N, K)"""
        y = np.atleast_2d(x) + 1.0
        return np.prod(np.cos(y[:, None, :] * self.w[None]), axis=2) / self.h

    def grad(self, x):
        """one point (d,) -> (K, d)"""
        c, s = np.cos((x + 1.0) * self.w), np.sin((x + 1.0) * self.w)  # (K, d)
        g = np.empty_like(c)
        for i in range(self.d):
            g[:, i] = -self.w[:, i] * s[:, i] * np.prod(np.delete(c, i, axis=1), axis=1)
        return g / self.h[:, None]


class TargetDensity:
    """
    Pluggable target density over the normalized wrist box. update(x, wrench) adds a visited sample; values(z) gives the
    unnormalized density at evaluation points z (N, d). Samples live in a ring buffer of @memory (kept across episodes).
    """

    def __init__(self, d, f_scale=5.0, bandwidth=0.15, memory=2000, kappa=1.0):
        self.d, self.f_scale, self.h, self.kappa = d, float(f_scale), float(bandwidth), float(kappa)
        self.X = np.zeros((memory, d))
        self.f = np.zeros(memory)
        self.n = 0

    def update(self, x, wrench):
        i = self.n % len(self.X)
        self.X[i], self.f[i] = x, float(np.linalg.norm(np.asarray(wrench)[3:]))  # |force| (N)
        self.n += 1

    def _kernel(self, z):
        m = min(self.n, len(self.X))
        d2 = ((z[:, None, :] - self.X[None, :m]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / self.h**2), self.f[:m]  # (N, m), (m,)

    def values(self, z):
        raise NotImplementedError


class UniformDensity(TargetDensity):
    def values(self, z):
        return np.ones(len(z))


class LowForceDensity(TargetDensity):
    """phi ~ exp(-f_hat^2 / f_scale^2), f_hat^2 a kernel regression of |f|^2 shrunk toward 0 (kappa pseudo-samples)"""

    def values(self, z):
        if self.n == 0:
            return np.ones(len(z))
        W, f = self._kernel(z)
        f2 = (W @ f**2) / (W.sum(1) + self.kappa)
        return np.exp(-f2 / self.f_scale**2)


class ForceVarianceDensity(TargetDensity):
    """phi ~ local variance of |f| (kernel weighted), shrunk toward f_scale^2 where little was visited, plus a floor"""

    def values(self, z):
        if self.n == 0:
            return np.ones(len(z))
        W, f = self._kernel(z)
        s = W.sum(1)
        mean = (W @ f) / np.maximum(s, 1e-12)
        var = (W * (f[None] - mean[:, None]) ** 2).sum(1)
        var = (var + self.kappa * self.f_scale**2) / (s + self.kappa)
        return var / self.f_scale**2 + 1e-3


DENSITIES = {"low_force": LowForceDensity, "force_variance": ForceVarianceDensity, "uniform": UniformDensity}


class ErgodicSampler(WristSampler):
    """Spectral multiscale coverage on @ergodic_axes (positions in self.axes = the active axes); see the module doc"""

    def __init__(self, axes, ergodic_axes=None, modes=5, umax=0.1, target="low_force", f_scale=5.0, bandwidth=0.15,
                 memory=2000, update_every=20, eval_points=4096, seed=0):  # fmt: skip
        super().__init__(axes)
        eax = list(self.axes) if ergodic_axes is None else [AXES.index(a) if a in AXES else int(a) for a in ergodic_axes]
        assert set(eax) <= set(self.axes), f"ergodic axes {eax} must be among the active wrist axes {list(self.axes)}"
        self.idx = np.array([list(self.axes).index(a) for a in eax])  # positions in the active-axes vector
        self.eax = np.array(eax)
        d = len(eax)
        self.basis = CosineBasis(d, modes)
        if self.basis.K > 5000:
            warnings.warn(f"ergodic: {self.basis.K} Fourier coefficients ({modes}^{d}); lower --ergodic_modes")
        self.umax, self.update_every = float(umax), int(update_every)
        self.density = DENSITIES[target](d, f_scale, bandwidth, memory)
        if d <= 2:
            g = np.linspace(-1, 1, 32)
            self.Z = np.array(np.meshgrid(*[g] * d, indexing="ij")).reshape(d, -1).T
        else:
            self.Z = np.random.default_rng(seed).uniform(-1, 1, (eval_points, d))
        self._FZ = self.basis.F(self.Z) if len(self.Z) * self.basis.K <= 4e6 else None
        self.phi_k = self._phi()
        self.n_obs = 0
        self.reset_trajectory()

    def _phi(self):
        """phi_k = int phi F_k with phi normalized: mean(phi F_k) / mean(phi) over the uniform evaluation points"""
        v = self.density.values(self.Z)
        if self._FZ is not None:
            return (v @ self._FZ) / v.sum()
        out = np.zeros(self.basis.K)
        for i in range(0, len(self.Z), 512):
            out += v[i:i + 512] @ self.basis.F(self.Z[i:i + 512])
        return out / v.sum()

    def reset_trajectory(self):
        self.sum_F = np.zeros(self.basis.K)
        self.t = 0.0

    def reset(self, env, rng):
        super().reset(env, rng)
        self.reset_trajectory()

    def observe(self, x, wrench, dt):
        """Adds the visited point @x (ergodic axes) with the wrench measured there"""
        self.sum_F += self.basis.F(x)[0] * dt
        self.t += dt
        self.density.update(x, wrench)
        self.n_obs += 1
        if self.n_obs % self.update_every == 0:
            self.phi_k = self._phi()

    def metric(self):
        """sum_k Lambda_k (c_k - phi_k)^2 of the current episode"""
        if self.t <= 0:
            return np.nan
        return float(self.basis.Lambda @ (self.sum_F / self.t - self.phi_k) ** 2)

    def feedback(self, x):
        x = np.clip(x, -0.99, 0.99)  # the cosine basis' gradient vanishes on the box's faces
        S = self.sum_F - self.t * self.phi_k
        B = (self.basis.Lambda * S) @ self.basis.grad(x)
        nb = np.linalg.norm(B)
        return np.zeros_like(x) if nb < 1e-12 else -self.umax * B / nb

    def sample(self, env, obs_info, dt):
        prev = obs_info.get("wrist_exec_prev")
        if prev is not None:
            self.observe(np.asarray(prev)[self.eax], obs_info["wrench"], dt)
        out = np.zeros(self.d)
        out[self.idx] = self.feedback(np.asarray(obs_info["wrist_cmd_agent"])[self.eax])
        return out


# ---- factory ---------------------------------------------------------------------------------------------------------
def _get(args, name):
    v = getattr(args, name, None)
    return DEFAULTS[name] if v is None and DEFAULTS[name] is not None else v


def flags_cov(args):
    """6x6 Sigma from --noise_cov, or --noise_std (+ --noise_corr)"""
    if _get(args, "noise_cov"):
        S = np.load(_get(args, "noise_cov"))
        assert S.shape == (6, 6), S.shape
        return S
    std = _vec6(_get(args, "noise_std"), "noise_std")
    C = np.load(_get(args, "noise_corr")) if _get(args, "noise_corr") else np.eye(6)
    assert C.shape == (6, 6), C.shape
    return std[:, None] * C * std[None, :]


def make_sampler(args, active_axes):
    """The sampler for --wrist_noise on @active_axes (None for none). @args: absl FLAGS, argparse namespace, ..."""
    kind = _get(args, "wrist_noise")
    assert kind in WRIST_NOISE, f"unknown wrist_noise {kind}"
    if kind == "none":
        return None
    if kind == "random":
        return RandomSampler(active_axes, _get(args, "noise_std"), _get(args, "noise_dist"))
    if kind == "gaussian":
        if _get(args, "noise_cov_source") == "force_budget":
            fb = dict(f_max=_get(args, "noise_f_max"), tau_max=_get(args, "noise_tau_max"), budget=_get(args, "noise_budget"),
                      caps=_vec6(_get(args, "noise_caps"), "noise_caps"), every=_get(args, "noise_cov_every"))  # fmt: skip
            return GaussianSampler(active_axes, force_budget=fb, tau=_get(args, "noise_tau"))
        return GaussianSampler(active_axes, cov=flags_cov(args), tau=_get(args, "noise_tau"))
    return ErgodicSampler(
        active_axes, _get(args, "ergodic_axes") or None, _get(args, "ergodic_modes"), _get(args, "ergodic_umax"),
        _get(args, "ergodic_target"), _get(args, "ergodic_f_scale"), _get(args, "ergodic_bandwidth"),
        _get(args, "ergodic_memory"), _get(args, "ergodic_update"),
    )  # fmt: skip


def noise_tag(args):
    """Short name of the exploration setting for run names, e.g. gaussian-fb-tau0.2"""
    kind = _get(args, "wrist_noise")
    if kind == "random":
        return f"random-{_get(args, 'noise_dist')[0]}{'_'.join(f'{float(s):g}' for s in np.atleast_1d(_get(args, 'noise_std')))}"
    if kind == "gaussian":
        src = "fb" if _get(args, "noise_cov_source") == "force_budget" else "cov" if _get(args, "noise_cov") else "std"
        tau = float(_get(args, "noise_tau"))
        return f"gaussian-{src}" + (f"-tau{tau:g}" if tau > 0 else "")
    if kind == "ergodic":
        return f"ergodic-{_get(args, 'ergodic_target')}-u{float(_get(args, 'ergodic_umax')):g}"
    return "none"


# ---- self-checks -----------------------------------------------------------------------------------------------------
class _FakeWrist:
    def __init__(self):
        from robosuite.models.wrists import StewartPlatformWrist

        self.w = StewartPlatformWrist()
        self.action_scale = self.w.action_scale

    def platform_stiffness(self, pose):
        return self.w.platform_stiffness(pose)


class _FakeEnv:
    def __init__(self):
        self.wrist = _FakeWrist()


def _check():
    rng = np.random.default_rng(0)
    dt = 0.05
    # 1. white and OU gaussian: empirical covariance vs Sigma; OU lag-1 autocorrelation vs exp(-dt / tau)
    A = rng.normal(size=(6, 6))
    Sigma = 0.01 * (A @ A.T + np.eye(6))
    for tau in (0.0, 0.2):
        s = GaussianSampler(range(6), cov=Sigma, tau=tau)
        s.reset(None, np.random.default_rng(1))
        X = np.array([s.sample(None, {}, dt) for _ in range(200000)])
        err = np.abs(np.cov(X.T) - Sigma).max() / np.abs(Sigma).max()
        ac = np.mean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in range(6)])
        print(f"gaussian tau {tau}: max |cov - Sigma| / max |Sigma| = {err:.3f}, lag-1 autocorr {ac:.3f} "
              f"(expected {np.exp(-dt / tau) if tau else 0.0:.3f})")  # fmt: skip
        assert err < 0.05 and abs(ac - (np.exp(-dt / tau) if tau else 0.0)) < 0.02
    # restricted to active axes: the marginal
    s = GaussianSampler((2,), cov=Sigma)
    s.reset(None, rng)
    v = np.var([s.sample(None, {}, dt) for _ in range(50000)])
    print(f"gaussian on z only: var {v:.5f} vs Sigma_zz {Sigma[2, 2]:.5f}")
    assert abs(v / Sigma[2, 2] - 1) < 0.05
    # random
    r = RandomSampler((0, 2), [0.1, 0.2, 0.3, 0.1, 0.1, 0.1], "uniform")
    r.reset(None, rng)
    X = np.array([r.sample(None, {}, dt) for _ in range(20000)])
    print(f"random uniform: max |x| {np.abs(X).max(0).round(3)} (std flags 0.1, 0.3)")
    assert np.all(np.abs(X).max(0) <= [0.1, 0.3])

    # 2. force budget: softer axes get more variance; the soft (extended) wrist more than the stiff one
    env = _FakeEnv()
    fb = dict(f_max=5.0, tau_max=0.25, budget=1.0, caps=np.ones(6), every=1)
    out = {}
    for z in (-1.0, 0.0, 1.0):
        s = GaussianSampler(range(6), force_budget=fb)
        s.reset(env, rng)
        s.sample(env, {"wrist_cmd_agent": np.array([0, 0, z, 0, 0, 0])}, dt)
        out[z] = s.Sigma
        Kn = env.wrist.platform_stiffness(np.array([0, 0, z, 0, 0, 0]) * env.wrist.action_scale) @ np.diag(env.wrist.action_scale)
        Wf = np.diag([1 / 25.0] * 3 + [1 / 0.0625] * 3)
        load = np.trace(Kn.T @ Wf @ Kn @ s.Sigma)
        print(f"force budget z {z:+.0f}: std per axis {np.sqrt(np.diag(s.Sigma)).round(4)} (normalized), "
              f"tr(R Sigma) {load:.3f}, log det {np.linalg.slogdet(s.Sigma)[1]:.1f}")  # fmt: skip
        assert load <= 1.0 + 1e-6 and np.all(np.diag(s.Sigma) <= 1 + 1e-9)
    S0 = out[0.0]
    Kh = env.wrist.platform_stiffness(np.zeros(6))
    print(f"  home: x (k {Kh[0, 0]:.0f} N/m) var {S0[0, 0]:.2e} > z (k {Kh[2, 2]:.0f} N/m) var {S0[2, 2]:.2e}")
    assert S0[0, 0] > S0[2, 2]
    assert np.linalg.slogdet(out[1.0])[1] > np.linalg.slogdet(out[0.0])[1] > np.linalg.slogdet(out[-1.0])[1]
    s = GaussianSampler((2,), force_budget=fb)
    s.reset(env, rng)
    s.sample(env, {"wrist_cmd_agent": np.zeros(6)}, dt)
    print(f"  z only at home: std {np.sqrt(s.Sigma[0, 0]):.4f} = f_max / (k_zz z_range) = {5.0 / (Kh[2, 2] * 0.02):.4f}")

    # 3. ergodic: on a uniform target, a point moving with the SMC feedback lowers the ergodic metric over time
    for d in (1, 2):
        e = ErgodicSampler(range(d), modes=8 if d == 1 else 6, umax=1.0, target="uniform")
        e.reset(None, rng)
        x = np.full(d, 0.3)
        ms = []
        for t in range(4000):
            e.observe(x, np.zeros(6), dt)
            x = np.clip(x + dt * e.feedback(x), -0.99, 0.99)
            ms.append(e.metric())
        print(f"ergodic d={d}, uniform target: metric {ms[50]:.4f} (t=2.5 s) -> {ms[500]:.4f} (25 s) -> {ms[-1]:.5f} (200 s)")
        assert ms[-1] < 0.2 * ms[50] and ms[500] < ms[50]
    # the low-force density prefers low-force regions: force grows with x, density at -0.8 > at +0.8
    e = ErgodicSampler((0, 2), target="low_force", modes=5, f_scale=5.0)
    for _ in range(500):
        x = rng.uniform(-1, 1, 2)
        e.observe(x, np.array([0, 0, 0, 0, 0, 10.0 * (x[0] + 1)]), dt)
    v = e.density.values(np.array([[-0.8, 0.0], [0.8, 0.0]]))
    print(f"low-force density at x=-0.8 / +0.8: {v.round(3)}")
    assert v[0] > v[1]
    print("all exploration checks passed")


if __name__ == "__main__":
    _check()
