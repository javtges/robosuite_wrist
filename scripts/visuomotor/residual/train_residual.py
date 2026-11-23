#!/usr/bin/env python3
"""
SERL residual fine-tuning on DoorResidual-v0 (door_residual_env.py): a frozen base (diffusion policy or the scripted
door oracle) plus an off-policy SAC residual on the wrist (and optionally the arm), with wrist-side exploration
(exploration.py). The structure of key_insertion/train_serl.py: SERL's asynchronous actor / learner (agentlace),
make_sac_agent unchanged, RLPD-style 50/50 sampling when a demo buffer exists, eval CSV, wandb, final checkpoint.
Runs in the SERL venv (Python 3.10), from scripts/:

    python -m visuomotor.residual.train_residual --learner --structure wrist_z --base oracle --wrist_noise gaussian \\
        --noise_cov_source force_budget --noise_tau 0.2 --demo_rollouts 20 --seed 0
    python -m visuomotor.residual.train_residual --actor   --structure wrist_z --base oracle --wrist_noise gaussian \\
        --noise_cov_source force_budget --noise_tau 0.2 --seed 0          # second terminal, same env/noise flags
    python -m visuomotor.residual.train_residual --learner --structure arm_wrist_6d --base dp --base_ckpt ../runs/dp/x/latest.pt

Both processes take the same flags (they build the same env to get its spaces). Runs on one machine need different
--port. --nodebug logs to wandb: project $WANDB_PROJECT (default hsa-door-visuomotor), run name --exp_name (default
<structure>-<noise>-<base>-s<seed>), group the same name without the seed, so seeds of one setting group together.

Actor: steps at most --actor_hz env steps per second (default 20, the env's control rate). Exploration (--wrist_noise)
is applied inside the training env to the wrist part of the executed action; the replay buffer stores the EXECUTED
residual (info["residual_executed"]: executed action mapped back into the agent's action space, clipped), not the
sampled one. Evaluation (every --eval_period env steps and at the end; deterministic policy, no exploration) appends
to <run_root>/<exp_name>/eval.csv: success rate, episode length, peak flange force, over-force rate, mean wrist z,
return. At the end the actor saves the latest parameters to <run_root>/<exp_name>/final/.

Learner: at most --max_update_ratio updates per online transition, as train_serl. --demo_rollouts N rolls out the base
with a zero residual (no exploration) for N episodes before training and keeps them as the demo half of every batch
(residual action 0; saved to <run_root>/<exp_name>/demos.pkl).
"""

import csv
import os
import pickle
import sys
import time

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from flax.training import checkpoints
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.utils.launcher import make_replay_buffer, make_sac_agent, make_trainer_config
from serl_launcher.utils.timer_utils import Timer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from visuomotor.residual import door_residual_env as dre  # noqa: E402  (registers DoorResidual-v0)
from visuomotor.residual.exploration import DEFAULTS as NOISE, WRIST_NOISE, make_sampler, noise_tag  # noqa: E402

FLAGS = flags.FLAGS
# ---- env -------------------------------------------------------------------------------------------------------------
flags.DEFINE_enum("structure", "wrist_z", list(dre.STRUCTURES), "Residual structure (door_residual_env.STRUCTURES).")
flags.DEFINE_enum("base", "oracle", ["oracle", "dp"], "Frozen base: scripted door oracle or diffusion policy.")
flags.DEFINE_string("base_ckpt", None, "dp base: train_dp.py checkpoint.")
flags.DEFINE_string("base_schedule", "soft_insert_home_turn", "oracle base: door_study wrist schedule.")
flags.DEFINE_float("base_static_z", None, "oracle base: a static wrist z instead of the schedule.")
flags.DEFINE_float("base_max_dx", 1.5, "oracle base: aiming error across the slot, uniform +- (mm).")
flags.DEFINE_float("base_max_dyaw", 3.0, "oracle base: aiming error in yaw, uniform +- (deg).")
flags.DEFINE_string("base_device", "cpu", "dp base: torch device.")
flags.DEFINE_list("cylinder_friction", ["0.15"], "Lock friction (N m); several are cycled per episode.")
flags.DEFINE_list("wrist_scale", ["1.0"], "Wrist residual scale, normalized units, 1 or 6 values (x y z rx ry rz).")
flags.DEFINE_float("pos_scale", 3.0, "Arm residual position bound (mm).")
flags.DEFINE_float("rot_scale", 2.0, "Arm residual rotation bound (deg).")
flags.DEFINE_float("wrist_rate", 1.0, "Max executed wrist command change per second (normalized; 0: no limit). Default 1 = the expert's switch rate; without a limit an untrained agent jumps the plate and the force limit ends the episode at once.")
flags.DEFINE_boolean("z_compensation", False, "Move the arm target against the commanded plate travel of z residuals.")
flags.DEFINE_boolean("privileged_obs", False, "Add the key tip's pose in the lock frame and the lock / door angles.")
flags.DEFINE_boolean("phase_obs", False, "Add a one-hot of the env-derived task phase.")
flags.DEFINE_float("force_penalty", dre.FORCE_PENALTY, "Weight of the per-step force penalty (0: success reward only).")
flags.DEFINE_float("force_limit", dre.FORCE_LIMIT, "Flange force (N) that ends the episode as a failure.")
flags.DEFINE_enum("force_measure", "physics", ["physics", "step"], "Force for reward / limit: physics-step peak or step end.")
flags.DEFINE_integer("max_traj_length", 300, "Episode length (env steps at 20 Hz), after the skipped approach.")
flags.DEFINE_boolean("skip_approach", True, "Run the free-space approach with zero residual before each episode.")
flags.DEFINE_boolean("end_on_base_failure", True, "Truncate when the oracle base gives up.")
# ---- exploration (exploration.py) ------------------------------------------------------------------------------------
flags.DEFINE_enum("wrist_noise", "none", list(WRIST_NOISE), "Wrist-side exploration during training (not eval).")
flags.DEFINE_list("noise_std", [str(v) for v in NOISE["noise_std"]], "random / gaussian (flags): std, 1 or 6 values.")
flags.DEFINE_enum("noise_dist", NOISE["noise_dist"], ["gaussian", "uniform"], "random: distribution (uniform on +-std).")
flags.DEFINE_string("noise_corr", None, "gaussian (flags): 6x6 correlation matrix (.npy).")
flags.DEFINE_string("noise_cov", None, "gaussian (flags): 6x6 covariance (.npy), overrides noise_std.")
flags.DEFINE_enum("noise_cov_source", NOISE["noise_cov_source"], ["flags", "force_budget"], "gaussian: where Sigma comes from.")
flags.DEFINE_float("noise_f_max", NOISE["noise_f_max"], "force_budget: force tolerance (N) in W_f.")
flags.DEFINE_float("noise_tau_max", NOISE["noise_tau_max"], "force_budget: moment tolerance (N m) in W_f.")
flags.DEFINE_float("noise_budget", NOISE["noise_budget"], "force_budget: tr(R Sigma) budget.")
flags.DEFINE_list("noise_caps", [str(v) for v in NOISE["noise_caps"]], "force_budget: per-axis std caps (normalized), 1 or 6.")
flags.DEFINE_integer("noise_cov_every", NOISE["noise_cov_every"], "force_budget: recompute Sigma every this many steps.")
flags.DEFINE_float("noise_tau", NOISE["noise_tau"], "gaussian: OU time constant (s); 0 = white.")
flags.DEFINE_list("ergodic_axes", None, "ergodic: wrist axes (names or indices; default all active).")
flags.DEFINE_integer("ergodic_modes", NOISE["ergodic_modes"], "ergodic: cosine modes per dimension.")
flags.DEFINE_float("ergodic_umax", NOISE["ergodic_umax"], "ergodic: perturbation magnitude (normalized).")
flags.DEFINE_enum("ergodic_target", NOISE["ergodic_target"], ["low_force", "force_variance", "uniform"], "ergodic: target density.")
flags.DEFINE_float("ergodic_f_scale", NOISE["ergodic_f_scale"], "ergodic: force scale (N) of the target density.")
flags.DEFINE_float("ergodic_bandwidth", NOISE["ergodic_bandwidth"], "ergodic: kernel bandwidth (normalized).")
flags.DEFINE_integer("ergodic_memory", NOISE["ergodic_memory"], "ergodic: samples kept for the target density.")
flags.DEFINE_integer("ergodic_update", NOISE["ergodic_update"], "ergodic: steps between target-coefficient updates.")
# ---- SERL (as train_serl) --------------------------------------------------------------------------------------------
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("demo_rollouts", 0, "Learner: zero-residual base episodes as the 50/50 demo buffer (0: none).")
flags.DEFINE_integer("batch_size", 256, "Batch size per critic update (half demos, half online with demos).")
flags.DEFINE_integer("critic_actor_ratio", 8, "Critic updates per actor update (UTD).")
flags.DEFINE_integer("max_steps", 1000000, "Actor: env steps. Learner: updates.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")
flags.DEFINE_integer("random_steps", 0, "Actor: uniform random residuals for this many env steps first.")
flags.DEFINE_integer("training_starts", 1000, "Learner: online transitions before training starts.")
flags.DEFINE_float("max_update_ratio", 0.5, "Learner: at most this many updates per online transition (0: no cap).")
flags.DEFINE_integer("steps_per_update", 30, "Actor: env steps between parameter syncs.")
flags.DEFINE_integer("log_period", 10, "Learner: updates between wandb logs.")
flags.DEFINE_float("actor_hz", 20.0, "Actor: at most this many env steps per second (0: as fast as it runs).")
flags.DEFINE_integer("eval_period", 2000, "Actor: env steps between evaluations.")
flags.DEFINE_integer("eval_n_trajs", 10, "Episodes per evaluation.")
flags.DEFINE_boolean("learner", False, "Run the learner.")
flags.DEFINE_boolean("actor", False, "Run the actor.")
flags.DEFINE_string("ip", "localhost", "Learner's IP address (actor).")
flags.DEFINE_integer("port", 5588, "agentlace port (the broadcast port is port + 1).")
flags.DEFINE_integer("checkpoint_period", 5000, "Learner: updates between checkpoints (0: none).")
flags.DEFINE_string("run_root", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"), "Eval CSVs and checkpoints.")
flags.DEFINE_string("exp_name", None, "Run name (default <structure>-<noise>-<base>-s<seed>).")
flags.DEFINE_string("wandb_project", None, "wandb project (default $WANDB_PROJECT or hsa-door-visuomotor).")
flags.DEFINE_boolean("debug", True, "No wandb (--nodebug to log to wandb).")


def base_tag():
    if FLAGS.base == "dp":
        ckpt = os.path.normpath(FLAGS.base_ckpt or "none")
        return "dp-" + os.path.basename(os.path.dirname(ckpt)) + "_" + os.path.splitext(os.path.basename(ckpt))[0]
    return "oracle-" + (f"z{FLAGS.base_static_z:g}" if FLAGS.base_static_z is not None else FLAGS.base_schedule)


def group_name():
    return f"{FLAGS.structure}-{noise_tag(FLAGS)}-{base_tag()}"


def exp_name():
    return FLAGS.exp_name or f"{group_name()}-s{FLAGS.seed}"


def run_dir():
    path = os.path.join(FLAGS.run_root, exp_name())
    os.makedirs(path, exist_ok=True)
    return path


def make_env(seed):
    return gym.make(
        "DoorResidual-v0", structure=FLAGS.structure, base=FLAGS.base, base_ckpt=FLAGS.base_ckpt,
        base_schedule=FLAGS.base_schedule, base_static_z=FLAGS.base_static_z, base_max_dx=FLAGS.base_max_dx,
        base_max_dyaw=FLAGS.base_max_dyaw, cylinder_friction=[float(f) for f in FLAGS.cylinder_friction],
        wrist_scale=[float(s) for s in FLAGS.wrist_scale], pos_scale=FLAGS.pos_scale * 1e-3,
        rot_scale=np.radians(FLAGS.rot_scale), wrist_rate=FLAGS.wrist_rate, z_compensation=FLAGS.z_compensation, privileged_obs=FLAGS.privileged_obs,
        phase_obs=FLAGS.phase_obs, force_limit=FLAGS.force_limit, force_penalty=FLAGS.force_penalty,
        force_measure=FLAGS.force_measure, max_steps=FLAGS.max_traj_length, skip_approach=FLAGS.skip_approach,
        end_on_base_failure=FLAGS.end_on_base_failure, seed=seed, device=FLAGS.base_device,
    )  # fmt: skip


def evaluate(agent, env, n):
    """Deterministic policy (the mode of the action distribution), no exploration, @n episodes"""
    success, lengths, peaks, over, z_mean, returns = [], [], [], [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = truncated = False
        zs, ret, info = [], 0.0, {}
        while not (done or truncated):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=jax.device_put(obs), argmax=True)))
            obs, reward, done, truncated, info = env.step(action)
            zs.append(info["wrist_z"])
            ret += reward
        success.append(info["success"])
        lengths.append(env.unwrapped.steps)
        peaks.append(info["peak_force"])
        over.append(info["over_force"])
        z_mean.append(np.mean(zs))
        returns.append(ret)
    return dict(success=float(np.mean(success)), length=float(np.mean(lengths)), peak_force=float(np.mean(peaks)),
                over_force=float(np.mean(over)), wrist_z=float(np.mean(z_mean)), ret=float(np.mean(returns)))  # fmt: skip


def transition(obs, action, next_obs, reward, done, truncated):
    return dict(observations=obs, actions=np.asarray(action, dtype=np.float32), next_observations=next_obs,
                rewards=np.asarray(reward, dtype=np.float32), masks=1.0 - done, dones=done or truncated)  # fmt: skip


def actor(agent: SACAgent, data_store, env, sampling_rng):
    client = TrainerClient("actor_env", FLAGS.ip, make_trainer_config(FLAGS.port, FLAGS.port + 1), data_store, wait_for_server=True)

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    env.unwrapped.set_sampler(make_sampler(FLAGS, env.unwrapped.wrist_axes))  # training env only
    eval_env = make_env(FLAGS.seed + 10000)
    eval_path = os.path.join(run_dir(), "eval.csv")
    fields = ["env_step", "episodes", "wall_time", "success", "length", "peak_force", "over_force", "wrist_z", "ret"]
    if not os.path.exists(eval_path):
        with open(eval_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    obs, _ = env.reset()
    episodes, t0 = 0, time.time()
    next_step_time = time.time()
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="actor"):
        timer.tick("total")
        if FLAGS.actor_hz > 0:
            time.sleep(max(0.0, next_step_time - time.time()))
            next_step_time = max(next_step_time, time.time()) + 1.0 / FLAGS.actor_hz
        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(observations=jax.device_put(obs), seed=key, deterministic=False)
                actions = np.asarray(jax.device_get(actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            # the executed residual (after exploration and the stroke's clip), not the sampled one
            data_store.insert(transition(obs, info["residual_executed"], next_obs, reward, done, truncated))
            obs = next_obs
            if done or truncated:
                episodes += 1
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0 or step == FLAGS.max_steps - 1:
            with timer.context("eval"):
                metrics = evaluate(agent, eval_env, FLAGS.eval_n_trajs)
            row = dict(env_step=step, episodes=episodes, wall_time=time.time() - t0, **metrics)
            with open(eval_path, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(row)
            tqdm.tqdm.write(f"eval @ {step}: " + ", ".join(f"{k} {v:.3g}" for k, v in metrics.items()))
            client.request("send-stats", {"eval": metrics})

        timer.tock("total")
        if step % FLAGS.log_period == 0:
            client.request("send-stats", {"timer": timer.get_average_times()})

    client.stop()
    final_dir = os.path.abspath(os.path.join(run_dir(), "final"))
    checkpoints.save_checkpoint(final_dir, agent.state, step=FLAGS.max_steps, overwrite=True)
    print(f"saved the final policy to {final_dir}")


def interleave(online, demo, n):
    """Concatenates two batches so that each of the @n consecutive minibatches update_high_utd takes is half and half"""

    def cat(a, b):
        a = jnp.reshape(a, (n, -1) + a.shape[1:])
        b = jnp.reshape(b, (n, -1) + b.shape[1:])
        c = jnp.concatenate([a, b], axis=1)
        return jnp.reshape(c, (-1,) + c.shape[2:])

    return jax.tree_util.tree_map(cat, dict(online), dict(demo))


def make_logger():
    config = WandBLogger.get_default_config()
    config.update(dict(project=FLAGS.wandb_project or os.environ.get("WANDB_PROJECT", "hsa-door-visuomotor"),
                       exp_descriptor=exp_name(), group=group_name(), tag=[FLAGS.structure, noise_tag(FLAGS), base_tag()]))  # fmt: skip
    return WandBLogger(wandb_config=config, variant={}, debug=FLAGS.debug)


def learner(rng, agent: SACAgent, replay_buffer, demo_buffer, sharding):
    wandb_logger = make_logger()
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}

    server = TrainerServer(make_trainer_config(FLAGS.port, FLAGS.port + 1), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(total=FLAGS.training_starts, initial=len(replay_buffer), desc="Filling up replay buffer")
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.close()

    server.publish_network(agent.state.params)
    print("sent initial network to actor")

    n = FLAGS.critic_actor_ratio
    per_buffer = FLAGS.batch_size * n // (2 if demo_buffer is not None else 1)
    replay_iterator = replay_buffer.get_iterator(sample_args={"batch_size": per_buffer}, device=sharding.replicate())
    demo_iterator = None
    if demo_buffer is not None:
        demo_iterator = demo_buffer.get_iterator(sample_args={"batch_size": per_buffer}, device=sharding.replicate())

    checkpoint_dir = os.path.abspath(os.path.join(run_dir(), "checkpoints"))
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        with timer.context("wait_for_data"):
            while FLAGS.max_update_ratio > 0 and update_steps > FLAGS.max_update_ratio * len(replay_buffer):
                time.sleep(0.01)
        if update_steps % 1000 == 0:
            tqdm.tqdm.write(f"updates {update_steps}, online transitions {len(replay_buffer)}: "
                            f"{update_steps / max(len(replay_buffer), 1):.2f} updates per transition")  # fmt: skip
        with timer.context("sample_replay_buffer"):
            batch = next(replay_iterator)
            if demo_iterator is not None:
                batch = interleave(batch, next(demo_iterator), n)

        with timer.context("train"):
            agent, update_info = agent.update_high_utd(batch, utd_ratio=n)
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            checkpoints.save_checkpoint(checkpoint_dir, agent.state, step=update_steps, keep=20)

        update_steps += 1


def base_rollouts(env, n):
    """@n episodes of the base alone (zero residual, no exploration) -> transitions with action 0"""
    out, opened = [], 0
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for ep in range(n):
        obs, _ = env.reset()
        done = truncated = False
        info = {}
        while not (done or truncated):
            next_obs, reward, done, truncated, info = env.step(zero)
            out.append(transition(obs, zero, np.asarray(next_obs, dtype=np.float32), reward, done, truncated))
            obs = next_obs
        opened += bool(info.get("success"))
        print(f"  base rollout {ep}: {env.unwrapped.steps} steps, success {info.get('success')}, peak {info.get('peak_force', 0):.1f} N")
    print(f"base rollouts: {opened}/{n} opened, {len(out)} transitions")
    return out


def main(_):
    assert FLAGS.learner != FLAGS.actor, "Pass exactly one of --learner / --actor"
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % len(devices) == 0
    print(f"run {exp_name()} (group {group_name()}) -> {run_dir()}")

    rng = jax.random.PRNGKey(FLAGS.seed)
    env = make_env(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_agent(seed=FLAGS.seed, sample_obs=env.observation_space.sample(), sample_action=env.action_space.sample())
    agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding.replicate())

    if FLAGS.learner:
        replay_buffer = make_replay_buffer(env, capacity=FLAGS.replay_buffer_capacity, type="replay_buffer")
        demo_buffer = None
        if FLAGS.demo_rollouts > 0:
            transitions = base_rollouts(env, FLAGS.demo_rollouts)
            with open(os.path.join(run_dir(), "demos.pkl"), "wb") as f:
                pickle.dump(transitions, f)
            demo_buffer = make_replay_buffer(env, capacity=len(transitions), type="replay_buffer")
            for t in transitions:
                demo_buffer.insert(t)
            print(f"demo buffer: {len(demo_buffer)} zero-residual transitions")
        learner(jax.device_put(sampling_rng, sharding.replicate()), agent, replay_buffer, demo_buffer, sharding)
    else:
        data_store = QueuedDataStore(2000)
        actor(agent, data_store, env, jax.device_put(sampling_rng, sharding.replicate()))
        os._exit(0)  # agentlace's threads would otherwise keep the process alive


if __name__ == "__main__":
    app.run(main)
