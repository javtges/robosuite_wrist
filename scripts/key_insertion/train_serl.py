#!/usr/bin/env python3
"""
SERL training on KeyInsertionWrist-v0: SERL's asynchronous actor / learner SAC from state (examples/async_sac_state_sim)
plus RLPD-style 50/50 sampling from the oracle demos (as in examples/async_drq_sim with --demo_path). Runs in the SERL
venv (Python 3.10, see requirements_serl.txt), from scripts/:

    python -m key_insertion.train_serl --learner --mode variable --seed 0 --demo_path key_insertion/demos/variable_20.pkl
    python -m key_insertion.train_serl --actor   --mode variable --seed 0        # in a second terminal

or both at once: bash key_insertion/launch_serl.sh variable 0. Runs on the same machine need different --port.

The actor steps at most --actor_hz env steps per second (default 10, the policy rate: as on the real robot), so the
sim's speed doesn't set how much data the learner gets. The learner does at most --max_update_ratio updates per online
transition (default 0.5, i.e. 4 critic steps per transition), so runs on different hardware train on the same schedule
as long as the learner keeps up with that (check "updates per transition" in learner.log). The actor evaluates the
deterministic policy every --eval_period env steps and at its last step, and appends a row to
<run_root>/<mode>_seed<seed>/eval.csv (success rate, episode length, peak flange force, over-force rate, wrist z,
return); with --nodebug it also logs to wandb. At the end it saves its policy (the learner's latest parameters) to
<run_root>/<mode>_seed<seed>/final/ (make_ft_figure.py loads it).

Changes from SERL's example: our env (gym.make with mode / seed), the demo buffer and its 50/50 sampling, interleaving
the two halves so that every critic minibatch is 50/50 (SAC.update_high_utd splits its batch into consecutive
minibatches), the eval CSV, and agentlace ports from --port. The agent is make_sac_agent unchanged (2x256 MLPs, layer
norm, 10-critic ensemble, discount 0.99).
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
from serl_launcher.utils.launcher import make_replay_buffer, make_sac_agent, make_trainer_config, make_wandb_logger
from serl_launcher.utils.timer_utils import Timer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import key_insertion.serl_env  # noqa: E402,F401  (registers KeyInsertionWrist-v0)

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", "variable", ["stiff", "soft", "variable"], "Wrist mode (the policy being trained).")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("demo_path", None, "Oracle demos (collect_demos.py). None: no demos (plain SAC).")
flags.DEFINE_integer("max_traj_length", 100, "Episode length (policy steps at 10 Hz).")
flags.DEFINE_float("force_penalty", key_insertion.serl_env.FORCE_PENALTY, "Weight of the env's per-step force penalty (0: success reward only).")
flags.DEFINE_integer("batch_size", 256, "Batch size per critic update (half demos, half online with demos).")
flags.DEFINE_integer("critic_actor_ratio", 8, "Critic updates per actor update (UTD).")
flags.DEFINE_integer("max_steps", 1000000, "Actor: env steps. Learner: updates.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")
flags.DEFINE_integer("random_steps", 0, "Actor: uniform random actions for this many env steps first.")
flags.DEFINE_integer("training_starts", 1000, "Learner: online transitions before training starts.")
flags.DEFINE_float(
    "max_update_ratio", 0.5, "Learner: at most this many updates (each critic_actor_ratio critic steps) per online transition (0: no cap)."
)
flags.DEFINE_integer("steps_per_update", 30, "Actor: env steps between parameter syncs.")
flags.DEFINE_integer("log_period", 10, "Learner: updates between wandb logs.")
flags.DEFINE_float("actor_hz", 10.0, "Actor: at most this many env steps per second, as the real robot (0: as fast as it runs).")
flags.DEFINE_integer("eval_period", 2000, "Actor: env steps between evaluations.")
flags.DEFINE_integer("eval_n_trajs", 10, "Episodes per evaluation.")
flags.DEFINE_boolean("learner", False, "Run the learner.")
flags.DEFINE_boolean("actor", False, "Run the actor.")
flags.DEFINE_string("ip", "localhost", "Learner's IP address (actor).")
flags.DEFINE_integer("port", 5488, "agentlace port (the broadcast port is port + 1).")
flags.DEFINE_integer("checkpoint_period", 5000, "Learner: updates between checkpoints (0: none).")
flags.DEFINE_string("run_root", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"), "Eval CSVs and checkpoints.")
flags.DEFINE_boolean("debug", True, "No wandb (--nodebug to log to wandb).")


def run_dir():
    path = os.path.join(FLAGS.run_root, f"{FLAGS.mode}_seed{FLAGS.seed}")
    os.makedirs(path, exist_ok=True)
    return path


def make_env(seed):
    return gym.make(
        "KeyInsertionWrist-v0", mode=FLAGS.mode, seed=seed, max_steps=FLAGS.max_traj_length, force_penalty=FLAGS.force_penalty
    )


def evaluate(agent, env, n):
    """Deterministic policy (the mode of the action distribution), @n episodes. Returns a dict of metrics."""
    seated, lengths, peaks, over, z_mean, returns = [], [], [], [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = truncated = False
        peak, zs, ret = 0.0, [], 0.0
        while not (done or truncated):
            action = np.asarray(jax.device_get(agent.sample_actions(observations=jax.device_put(obs), argmax=True)))
            obs, reward, done, truncated, info = env.step(action)
            peak = max(peak, info["max_force"])
            zs.append(info["wrist_z"])
            ret += reward
        seated.append(info["succeed"])
        lengths.append(env.unwrapped.steps)
        peaks.append(peak)
        over.append(info["over_force"])
        z_mean.append(np.mean(zs))
        returns.append(ret)
    return dict(
        success=float(np.mean(seated)),
        length=float(np.mean(lengths)),
        peak_force=float(np.mean(peaks)),
        over_force=float(np.mean(over)),
        wrist_z=float(np.mean(z_mean)),
        ret=float(np.mean(returns)),
    )


def actor(agent: SACAgent, data_store, env, sampling_rng):
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(FLAGS.port, FLAGS.port + 1),
        data_store,
        wait_for_server=True,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

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
            # No catching up after a slow step (e.g. an evaluation): never faster than actor_hz
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
            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=np.asarray(reward, dtype=np.float32),
                    masks=1.0 - done,
                    dones=done or truncated,
                )
            )
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


def learner(rng, agent: SACAgent, replay_buffer, demo_buffer, sharding):
    wandb_logger = make_wandb_logger(
        project="serl_key_insertion",
        description=f"{FLAGS.mode}_seed{FLAGS.seed}",
        debug=FLAGS.debug,
    )
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
                            f"{update_steps / max(len(replay_buffer), 1):.2f} updates per transition")
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


def main(_):
    assert FLAGS.learner != FLAGS.actor, "Pass exactly one of --learner / --actor"
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % len(devices) == 0

    rng = jax.random.PRNGKey(FLAGS.seed)
    env = make_env(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )
    agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding.replicate())

    if FLAGS.learner:
        replay_buffer = make_replay_buffer(env, capacity=FLAGS.replay_buffer_capacity, type="replay_buffer")
        demo_buffer = None
        if FLAGS.demo_path:
            with open(FLAGS.demo_path, "rb") as f:
                transitions = pickle.load(f)
            demo_buffer = make_replay_buffer(env, capacity=len(transitions), type="replay_buffer")
            for t in transitions:
                demo_buffer.insert(t)
            print(f"demo buffer: {len(demo_buffer)} transitions from {FLAGS.demo_path}")
        learner(jax.device_put(sampling_rng, sharding.replicate()), agent, replay_buffer, demo_buffer, sharding)
    else:
        data_store = QueuedDataStore(2000)
        actor(agent, data_store, env, jax.device_put(sampling_rng, sharding.replicate()))
        os._exit(0)  # agentlace's threads would otherwise keep the process alive


if __name__ == "__main__":
    app.run(main)
