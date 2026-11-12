"""
Peak flange force per rollout, from HSA-UMI's rosbags, into the hardware ledger.

    cd scripts
    python -m arm_impedance.bag_forces ../../HSA-UMI/data/key_in_hole/wrist/v1/eval/noarducam_ep030 --out arm_impedance/hardware_ledgers/noarducam_ep030

One bag per session holds every trial of that session; rollout_ledger's rows carry `started_at` (local wall clock)
and `time_s`, so each trial is the window [started_at, started_at + time_s] of the bag's wrench stream
(`/franka_robot_state_broadcaster/external_wrench_in_base_frame`, libfranka's O_F_ext_hat_K). The Franka's
estimate has a bias of a few newtons, so each trial is zeroed with the median wrench over the 2 s before its start,
when the arm is still. The output ledger has the converted rows (hsa_ledger.convert) plus `peak_flange_N`, and the
two charts. Needs the `rosbags` package (pure Python, no ROS).
"""

import argparse
import csv
import datetime
import os
import sys
import zoneinfo

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_impedance.hsa_ledger import convert  # noqa: E402
from arm_impedance.ledger_figures import LEDGER_COLUMNS, make_ledger_figures, summarize  # noqa: E402

WRENCH_TOPIC = "/franka_robot_state_broadcaster/external_wrench_in_base_frame"
CHUNK_TOPIC = "/polyumi/target_poses_traj"
TZ = zoneinfo.ZoneInfo("America/Chicago")


def read_streams(bag_dir):
    """(wrench times s, wrench |F| rows (n, 3) N, chunk times s) from one rosbag2 directory"""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    store = get_typestore(Stores.ROS2_HUMBLE)
    t_w, f_w, t_c = [], [], []
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic in (WRENCH_TOPIC, CHUNK_TOPIC)]
        for conn, t_ns, raw in reader.messages(connections=conns):
            if conn.topic == CHUNK_TOPIC:
                t_c.append(t_ns * 1e-9)
                continue
            msg = store.deserialize_cdr(raw, conn.msgtype)
            t_w.append(t_ns * 1e-9)
            f_w.append((msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z))
    return np.array(t_w), np.array(f_w), np.array(t_c)


def trial_window(row):
    start = datetime.datetime.strptime(row["started_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()
    return start, start + float(row["time_s"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_dir", help="an HSA-UMI eval directory with ledger.csv and bags/<name>/")
    p.add_argument("--out", required=True)
    p.add_argument("--zero-s", type=float, default=2.0, help="seconds before each trial used to zero the wrench")
    args = p.parse_args()

    root = os.path.normpath(os.path.join(args.eval_dir, "..", "..", "..", ".."))
    with open(os.path.join(args.eval_dir, "ledger.csv"), newline="") as f:
        raw_rows = [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
    rows = convert(os.path.join(args.eval_dir, "ledger.csv"), root)
    streams = {}
    for raw, row in zip(raw_rows, rows):
        bag = raw.get("bag", "").strip()
        bag_dir = os.path.join(args.eval_dir, "bags", bag)
        if not bag or not os.path.isdir(bag_dir):
            row["notes"] += " [no bag]"
            continue
        if bag not in streams:
            print(f"reading {bag_dir} ...", flush=True)
            streams[bag] = read_streams(bag_dir)
        t_w, f_w, t_c = streams[bag]
        t0, t1 = trial_window(raw)
        pre = (t_w >= t0 - args.zero_s) & (t_w < t0)
        win = (t_w >= t0) & (t_w <= t1)
        chunks = int(((t_c >= t0) & (t_c <= t1)).sum())
        if not win.any() or not pre.any():
            row["notes"] += f" [no wrench samples in window; bag spans {t_w[0] - t0:+.0f} .. {t_w[-1] - t0:+.0f} s from the start]"
            continue
        zero = np.median(f_w[pre], axis=0)
        mag = np.linalg.norm(f_w[win] - zero, axis=1)
        row["peak_flange_N"] = float(mag.max())
        row["notes"] += f" [{win.sum()} wrench samples, {chunks} chunks in window, bias {np.linalg.norm(zero):.1f} N]"
        print(f"  trial {raw['rollout']:>3s} {row['result']:7s} window {t1 - t0:5.1f} s: peak {mag.max():5.1f} N, {chunks} chunks")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summarize(rows, title=f"{args.eval_dir}: peak flange force from the bags"))
    print(open(os.path.join(args.out, "summary.md")).read())
    make_ledger_figures(os.path.join(args.out, "episodes.csv"), args.out)


if __name__ == "__main__":
    main()
