
"""
Helper CLI for bandit-style autoresearch.

Examples:
  python bandit_controller.py init --results results.tsv
  python bandit_controller.py next-arm --results results.tsv
  python bandit_controller.py append --results results.tsv --log run.log --commit abc1234 --arm optimizer --description "lower matrix lr"
  python bandit_controller.py frontier --results results.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ARMS = [
    "retest_frontier",
    "optimizer",
    "schedule",
    "shape",
    "attention_rope",
    "residual_ve",
    "mlp_block",
    "batch_efficiency",
]

HEADER = [
    "experiment",
    "commit",
    "arm",
    "val_bpb",
    "memory_gb",
    "tok_per_sec",
    "mfu_percent",
    "num_params_m",
    "flops_per_token_g",
    "reward",
    "status",
    "description",
]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def read_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_header(path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()


def append_row(path: Path, row: dict):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def first_success(rows):
    for row in rows:
        if row["status"] != "crash":
            return row
    return None


def dominates(a, b):
    if a["status"] == "crash" or b["status"] == "crash":
        return False
    return (
        float(a["val_bpb"]) <= float(b["val_bpb"])
        and float(a["tok_per_sec"]) >= float(b["tok_per_sec"])
        and float(a["memory_gb"]) <= float(b["memory_gb"])
        and float(a["flops_per_token_g"]) <= float(b["flops_per_token_g"])
        and (
            float(a["val_bpb"]) < float(b["val_bpb"])
            or float(a["tok_per_sec"]) > float(b["tok_per_sec"])
            or float(a["memory_gb"]) < float(b["memory_gb"])
            or float(a["flops_per_token_g"]) < float(b["flops_per_token_g"])
        )
    )


def compute_reward(summary, baseline):
    if baseline is None:
        quality = speed = memory = cost = 0.5
    else:
        quality = clamp(0.5 + (float(baseline["val_bpb"]) - float(summary["val_bpb"])) / 0.010)
        speed = clamp(
            0.5
            + (float(summary["steady_tok_per_sec"]) - float(baseline["tok_per_sec"]))
            / (0.25 * max(float(baseline["tok_per_sec"]), 1e-9))
        )
        memory = clamp(
            0.5
            + (float(baseline["memory_gb"]) - float(summary["peak_vram_gb"]))
            / (0.25 * max(float(baseline["memory_gb"]), 1e-9))
        )
        cost = clamp(
            0.5
            + (float(baseline["flops_per_token_g"]) - float(summary["num_flops_per_token_G"]))
            / (0.25 * max(float(baseline["flops_per_token_g"]), 1e-9))
        )
    reward = 0.45 * quality + 0.20 * speed + 0.15 * memory + 0.10 * cost + 0.10 * 1.0
    return reward


def parse_summary_line(log_path: Path):
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text().splitlines()):
        if line.startswith("run_summary_json: "):
            return json.loads(line.split("run_summary_json: ", 1)[1].strip())
    return None


def cmd_init(args):
    write_header(args.results)
    print(f"Initialized {args.results}")


def cmd_next_arm(args):
    rows = read_rows(args.results)
    stats = defaultdict(lambda: {"n": 0, "reward_sum": 0.0, "last_seen": -10**9, "recent_crashes": 0})
    crash_streak = defaultdict(int)
    for i, row in enumerate(rows, start=1):
        arm = row["arm"]
        stats[arm]["n"] += 1
        stats[arm]["reward_sum"] += float(row["reward"])
        stats[arm]["last_seen"] = i
        if row["status"] == "crash":
            crash_streak[arm] += 1
        else:
            crash_streak[arm] = 0
        stats[arm]["recent_crashes"] = crash_streak[arm]

    for arm in ARMS:
        if stats[arm]["n"] == 0:
            print(arm)
            return

    t = max(1, len(rows))
    best_arm, best_score = None, -1e9
    for arm in ARMS:
        n = stats[arm]["n"]
        mean_reward = stats[arm]["reward_sum"] / n
        score = mean_reward + 0.35 * math.sqrt(math.log(t + 1) / n)

        if (t + 1) % 8 == 0 and arm == "retest_frontier":
            score += 1.0
        if t - stats[arm]["last_seen"] >= 10:
            score += 0.05
        if stats[arm]["recent_crashes"] >= 2:
            score -= 0.15

        if score > best_score:
            best_arm, best_score = arm, score

    print(best_arm)


def cmd_append(args):
    rows = read_rows(args.results)
    summary = parse_summary_line(args.log)
    experiment = len(rows) + 1

    if summary is None:
        row = {
            "experiment": experiment,
            "commit": args.commit,
            "arm": args.arm,
            "val_bpb": "0.000000",
            "memory_gb": "0.0",
            "tok_per_sec": "0.0",
            "mfu_percent": "0.0",
            "num_params_m": "0.0",
            "flops_per_token_g": "0.0",
            "reward": "0.0000",
            "status": "crash",
            "description": args.description,
        }
    else:
        baseline = first_success(rows)
        reward = compute_reward(summary, baseline)
        row = {
            "experiment": experiment,
            "commit": args.commit,
            "arm": summary.get("run_arm") or args.arm,
            "val_bpb": f"{float(summary['val_bpb']):.6f}",
            "memory_gb": f"{float(summary['peak_vram_gb']):.3f}",
            "tok_per_sec": f"{float(summary['steady_tok_per_sec']):.1f}",
            "mfu_percent": f"{float(summary['mfu_percent']):.2f}",
            "num_params_m": f"{float(summary['num_params_M']):.1f}",
            "flops_per_token_g": f"{float(summary['num_flops_per_token_G']):.3f}",
            "reward": f"{reward:.4f}",
            "status": "keep",
            "description": summary.get("run_note") or args.description,
        }

        candidate = row
        dominated = False
        for other in rows:
            if dominates(other, candidate):
                dominated = True
                break
        if dominated:
            row["status"] = "discard"

    append_row(args.results, row)
    print(json.dumps(row, indent=2))


def cmd_frontier(args):
    rows = read_rows(args.results)
    frontier = []
    for i, row in enumerate(rows):
        if row["status"] == "crash":
            continue
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if dominates(other, row):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda r: float(r["val_bpb"]))
    print(json.dumps(frontier, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Bandit controller helpers")
    sub = parser.add_subparsers(required=True)

    p_init = sub.add_parser("init", help="Initialize results.tsv")
    p_init.add_argument("--results", type=Path, required=True)
    p_init.set_defaults(func=cmd_init)

    p_next = sub.add_parser("next-arm", help="Choose the next arm via UCB")
    p_next.add_argument("--results", type=Path, required=True)
    p_next.set_defaults(func=cmd_next_arm)

    p_append = sub.add_parser("append", help="Append a run from run.log")
    p_append.add_argument("--results", type=Path, required=True)
    p_append.add_argument("--log", type=Path, required=True)
    p_append.add_argument("--commit", type=str, required=True)
    p_append.add_argument("--arm", type=str, required=True)
    p_append.add_argument("--description", type=str, required=True)
    p_append.set_defaults(func=cmd_append)

    p_frontier = sub.add_parser("frontier", help="Print the current Pareto frontier")
    p_frontier.add_argument("--results", type=Path, required=True)
    p_frontier.set_defaults(func=cmd_frontier)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
