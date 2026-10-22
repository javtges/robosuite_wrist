"""
Writes teleop demonstrations to one hdf5 file that both robomimic and robosuite's playback script read.

    data                                    attrs: total, env_args (robomimic), env + env_info + repository_version +
                                            date + time (robosuite's playback_demonstrations_from_hdf5.py),
                                            wrist_config (the StewartPlatformConfig the demos were recorded with)
      demo_<i>                              attrs: num_samples, model_file (MJCF), success, success_any, plan_row,
                                            condition, input, start_time, duration_s, and the controller state a
                                            replay must restore: gripper_current_action, controller_initial_joint
        states        (N, S)                flattened MuJoCo state BEFORE each action
        actions       (N, 13)               the absolute teleop target: tcp pos (3) + rotvec (3), base frame;
                                            wrist pose (6, normalized); gripper (1)
        rewards, dones (N,)                 dones[-1] = 1
        obs/<key>, next_obs/<key>           every non-image observation
        teleop/stage  (N,)                  HSA-UMI's stage counter (Space)
        teleop/mode   (N,)                  what the input drove: 0 arm, 1 wrist, 2 nullspace (the wrist, with the arm
                                            cancelling it; see nullspace/README.md)
        teleop/wrist_stiffness_scale (N,)   leg stiffness multiplier (x toggles it). Not part of the MuJoCo state:
                                            a replay must set it per step (validate_demos.py does)

Images are not stored; robomimic's dataset_states_to_obs.py renders them from the states.
Restarting with the same file appends demos. Each demo is written and flushed when it is saved.
"""

import datetime
import json
import time

import h5py
import numpy as np


class DemoWriter:
    def __init__(self, path, env_args, wrist_config):
        self.path = path
        self.f = h5py.File(path, "a")
        data = self.f.require_group("data")
        env_args_json = json.dumps(env_args)
        if "env_args" in data.attrs:
            if json.loads(data.attrs["env_args"]) != env_args:
                self.f.close()
                raise ValueError(f"{path} holds demos of a different env / controller setup; use another --out")
        else:
            now = datetime.datetime.now()
            env_info = dict(env_args["env_kwargs"], env_name=env_args["env_name"])
            data.attrs.update(
                total=0,
                env_args=env_args_json,
                env=env_args["env_name"],
                env_info=json.dumps(env_info),
                repository_version=env_args["env_version"],
                date=now.strftime("%m-%d-%Y"),
                time=now.strftime("%H:%M:%S"),
                wrist_config=json.dumps(wrist_config),
            )
        self.data = data
        self.recording = False
        self._steps = []

    @property
    def demo_names(self):
        return sorted(self.data.keys(), key=lambda k: int(k.split("_")[1]))

    def start(self, attrs):
        self.recording = True
        self._steps = []
        self._attrs = dict(attrs, start_time=time.time())

    def add(self, state, action, reward, obs, next_obs, stage, mode, stiffness_scale):
        self._steps.append((state, action, reward, obs, next_obs, stage, mode, stiffness_scale))

    @property
    def num_steps(self):
        return len(self._steps)

    def discard(self):
        self.recording = False
        self._steps = []

    def save(self, success, success_any):
        """Writes the episode in progress. Returns the demo name, or None if it is empty."""
        self.recording = False
        if not self._steps:
            return None
        states, actions, rewards, obs, next_obs, stage, mode, scale = zip(*self._steps)
        names = self.demo_names
        name = f"demo_{int(names[-1].split('_')[1]) + 1 if names else 0}"
        g = self.data.create_group(name)
        n = len(states)
        g.create_dataset("states", data=np.array(states))
        g.create_dataset("actions", data=np.array(actions))
        g.create_dataset("rewards", data=np.array(rewards, dtype=np.float64))
        dones = np.zeros(n, dtype=np.int64)
        dones[-1] = 1
        g.create_dataset("dones", data=dones)
        for group, seq in (("obs", obs), ("next_obs", next_obs)):
            for k in seq[0]:
                g.create_dataset(f"{group}/{k}", data=np.array([o[k] for o in seq]))
        g.create_dataset("teleop/stage", data=np.array(stage, dtype=np.int64))
        g.create_dataset("teleop/mode", data=np.array(mode, dtype=np.int64))
        g.create_dataset("teleop/wrist_stiffness_scale", data=np.array(scale, dtype=np.float64))
        attrs = dict(self._attrs, num_samples=n, success=bool(success), success_any=bool(success_any))
        attrs["duration_s"] = time.time() - attrs["start_time"]
        for k, v in attrs.items():
            g.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        self.data.attrs["total"] = int(self.data.attrs["total"]) + n
        self.f.flush()
        self._steps = []
        return name

    def delete_last(self):
        """Deletes the last saved demo. Returns its name, or None if there is none."""
        names = self.demo_names
        if not names:
            return None
        n = int(self.data[names[-1]].attrs["num_samples"])
        del self.data[names[-1]]
        self.data.attrs["total"] = int(self.data.attrs["total"]) - n
        self.f.flush()
        return names[-1]

    def close(self):
        self.f.close()
