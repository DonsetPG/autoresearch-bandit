"""
Helper CLI for objective-arm bandit autoresearch.

In this version, the bandit arms are *objectives* rather than code areas.
The agent may modify code anywhere in the repo, but each experiment is tagged
with one primary objective arm such as quality, speed, or memory.

Examples:
  python bandit_controller.py init --results results.tsv
  python bandit_controller.py next-plan --results results.tsv
  python bandit_controller.py next-arm --results results.tsv
  python bandit_controller.py append \
      --results results.tsv \
      --log run.log \
      --commit abc1234 \
      --parent-commit def5678 \
      --parent-experiment 12 \
      --objective-arm speed \
      --description "improve fused optimizer step"
  python bandit_controller.py frontier --results results.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

OBJECTIVE_ARMS = ["quality", "speed", "memory", "params", "cost"]
OBJECTIVE_WEIGHTS = {
    "quality": 0.40,
    "speed": 0.20,
    "memory": 0.15,
    "params": 0.10,
    "cost": 0.15,
}
PRIMARY_ARM_WEIGHT = 0.65
GLOBAL_WEIGHT_IN_ARM_REWARD = 0.35
QUALITY_BAND = 0.010
RELATIVE_BAND = 0.25
UCB_EXPLORATION = 0.35

HEADER = [
    "experiment",
    "parent_experiment",
    "parent_commit",
    "commit",
    "objective_arm",
    "val_bpb",
    "tok_per_sec",
    "memory_gb",
    "num_params_m",
    "flops_per_token_g",
    "mfu_percent",
    "startup_seconds",
    "training_seconds",
    "total_seconds",
    "quality_score",
    "speed_score",
    "memory_score",
    "params_score",
    "cost_score",
    "global_reward",
    "arm_reward",
    "status",
    "description",
]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def parse_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def parse_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def is_success(row: Dict[str, Any]) -> bool:
    return row.get("status") != "crash"


def field_or(summary: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in summary and summary[name] not in (None, ""):
            return summary[name]
    return default


def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if not is_success(a) or not is_success(b):
        return False
    no_worse = (
        a["val_bpb"] <= b["val_bpb"]
        and a["tok_per_sec"] >= b["tok_per_sec"]
        and a["memory_gb"] <= b["memory_gb"]
        and a["num_params_m"] <= b["num_params_m"]
        and a["flops_per_token_g"] <= b["flops_per_token_g"]
    )
    strictly_better = (
        a["val_bpb"] < b["val_bpb"]
        or a["tok_per_sec"] > b["tok_per_sec"]
        or a["memory_gb"] < b["memory_gb"]
        or a["num_params_m"] < b["num_params_m"]
        or a["flops_per_token_g"] < b["flops_per_token_g"]
    )
    return no_worse and strictly_better


def first_success(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in rows:
        if is_success(row):
            return row
    return None


def compute_scores(row: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not is_success(row):
        return {
            "quality_score": 0.0,
            "speed_score": 0.0,
            "memory_score": 0.0,
            "params_score": 0.0,
            "cost_score": 0.0,
            "global_reward": 0.0,
            "arm_reward": 0.0,
        }
    if baseline is None:
        return {
            "quality_score": 0.5,
            "speed_score": 0.5,
            "memory_score": 0.5,
            "params_score": 0.5,
            "cost_score": 0.5,
            "global_reward": 0.5,
            "arm_reward": 0.5,
        }

    base_speed = max(baseline["tok_per_sec"], 1e-9)
    base_memory = max(baseline["memory_gb"], 1e-9)
    base_params = max(baseline["num_params_m"], 1e-9)
    base_cost = max(baseline["flops_per_token_g"], 1e-9)

    quality_score = clamp(0.5 + (baseline["val_bpb"] - row["val_bpb"]) / QUALITY_BAND)
    speed_score = clamp(0.5 + (row["tok_per_sec"] - baseline["tok_per_sec"]) / (RELATIVE_BAND * base_speed))
    memory_score = clamp(0.5 + (baseline["memory_gb"] - row["memory_gb"]) / (RELATIVE_BAND * base_memory))
    params_score = clamp(0.5 + (baseline["num_params_m"] - row["num_params_m"]) / (RELATIVE_BAND * base_params))
    cost_score = clamp(0.5 + (baseline["flops_per_token_g"] - row["flops_per_token_g"]) / (RELATIVE_BAND * base_cost))

    global_reward = (
        OBJECTIVE_WEIGHTS["quality"] * quality_score
        + OBJECTIVE_WEIGHTS["speed"] * speed_score
        + OBJECTIVE_WEIGHTS["memory"] * memory_score
        + OBJECTIVE_WEIGHTS["params"] * params_score
        + OBJECTIVE_WEIGHTS["cost"] * cost_score
    )

    objective_arm = row.get("objective_arm") or "quality"
    objective_score_name = f"{objective_arm}_score"
    objective_score = {
        "quality_score": quality_score,
        "speed_score": speed_score,
        "memory_score": memory_score,
        "params_score": params_score,
        "cost_score": cost_score,
    }.get(objective_score_name, quality_score)

    arm_reward = PRIMARY_ARM_WEIGHT * objective_score + GLOBAL_WEIGHT_IN_ARM_REWARD * global_reward

    return {
        "quality_score": quality_score,
        "speed_score": speed_score,
        "memory_score": memory_score,
        "params_score": params_score,
        "cost_score": cost_score,
        "global_reward": global_reward,
        "arm_reward": arm_reward,
    }


def coerce_row(row: Dict[str, Any]) -> Dict[str, Any]:
    objective_arm = row.get("objective_arm") or row.get("arm") or "quality"
    if objective_arm not in OBJECTIVE_ARMS:
        objective_arm = "quality"
    return {
        "experiment": parse_int(row.get("experiment"), 0),
        "parent_experiment": parse_int(row.get("parent_experiment"), 0),
        "parent_commit": row.get("parent_commit", ""),
        "commit": row.get("commit", ""),
        "objective_arm": objective_arm,
        "val_bpb": parse_float(row.get("val_bpb"), 0.0),
        "tok_per_sec": parse_float(row.get("tok_per_sec"), row.get("steady_tok_per_sec", 0.0)),
        "memory_gb": parse_float(row.get("memory_gb"), row.get("peak_vram_gb", 0.0)),
        "num_params_m": parse_float(row.get("num_params_m"), row.get("num_params_M", 0.0)),
        "flops_per_token_g": parse_float(row.get("flops_per_token_g"), row.get("num_flops_per_token_G", 0.0)),
        "mfu_percent": parse_float(row.get("mfu_percent"), row.get("mean_mfu_percent", 0.0)),
        "startup_seconds": parse_float(row.get("startup_seconds"), 0.0),
        "training_seconds": parse_float(row.get("training_seconds"), 0.0),
        "total_seconds": parse_float(row.get("total_seconds"), 0.0),
        "quality_score": parse_float(row.get("quality_score"), 0.0),
        "speed_score": parse_float(row.get("speed_score"), 0.0),
        "memory_score": parse_float(row.get("memory_score"), 0.0),
        "params_score": parse_float(row.get("params_score"), 0.0),
        "cost_score": parse_float(row.get("cost_score"), 0.0),
        "global_reward": parse_float(row.get("global_reward"), 0.0),
        "arm_reward": parse_float(row.get("arm_reward"), row.get("reward", 0.0)),
        "status": row.get("status", "discard"),
        "description": row.get("description", ""),
    }


def enrich_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [coerce_row(row) for row in rows]
    baseline = first_success(rows)
    if baseline is None:
        return rows
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated.update(compute_scores(updated, baseline))
        enriched.append(updated)
    return enriched


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    return enrich_rows(rows)


def write_header(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()


def append_row(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def pareto_indices(rows: List[Dict[str, Any]]) -> List[int]:
    success_indices = [i for i, row in enumerate(rows) if is_success(row)]
    frontier: List[int] = []
    for i in success_indices:
        dominated = False
        for j in success_indices:
            if i == j:
                continue
            if dominates(rows[j], rows[i]):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def frontier_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [rows[i] for i in pareto_indices(rows)]


def parse_summary_line(log_path: Path) -> Optional[Dict[str, Any]]:
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors="replace").splitlines()
    for line in reversed(lines):
        if line.startswith("run_summary_json: "):
            payload = line.split("run_summary_json: ", 1)[1].strip()
            return json.loads(payload)
    return None


def build_row_from_summary(
    summary: Dict[str, Any],
    experiment: int,
    commit: str,
    parent_commit: str,
    parent_experiment: int,
    default_objective_arm: str,
    description: str,
    existing_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    objective_arm = field_or(summary, "run_objective_arm", "run_arm", default=default_objective_arm) or default_objective_arm
    if objective_arm not in OBJECTIVE_ARMS:
        objective_arm = default_objective_arm

    row: Dict[str, Any] = {
        "experiment": experiment,
        "parent_experiment": parent_experiment,
        "parent_commit": parent_commit,
        "commit": commit,
        "objective_arm": objective_arm,
        "val_bpb": parse_float(field_or(summary, "val_bpb"), 0.0),
        "tok_per_sec": parse_float(field_or(summary, "steady_tok_per_sec", "tok_per_sec"), 0.0),
        "memory_gb": parse_float(field_or(summary, "peak_vram_gb", "memory_gb"), 0.0),
        "num_params_m": parse_float(field_or(summary, "num_params_M", "num_params_m"), 0.0),
        "flops_per_token_g": parse_float(field_or(summary, "num_flops_per_token_G", "flops_per_token_g"), 0.0),
        "mfu_percent": parse_float(field_or(summary, "mfu_percent", "mean_mfu_percent"), 0.0),
        "startup_seconds": parse_float(field_or(summary, "startup_seconds"), 0.0),
        "training_seconds": parse_float(field_or(summary, "training_seconds"), 0.0),
        "total_seconds": parse_float(field_or(summary, "total_seconds"), 0.0),
        "quality_score": 0.0,
        "speed_score": 0.0,
        "memory_score": 0.0,
        "params_score": 0.0,
        "cost_score": 0.0,
        "global_reward": 0.0,
        "arm_reward": 0.0,
        "status": "discard",
        "description": field_or(summary, "run_note", default=description) or description,
    }

    baseline = first_success(existing_rows)
    row.update(compute_scores(row, baseline))

    candidate_rows = existing_rows + [row]
    frontier = pareto_indices(candidate_rows)
    row["status"] = "keep" if (len(candidate_rows) - 1) in frontier else "discard"
    return row


def empty_crash_row(
    experiment: int,
    commit: str,
    parent_commit: str,
    parent_experiment: int,
    objective_arm: str,
    description: str,
) -> Dict[str, Any]:
    return {
        "experiment": experiment,
        "parent_experiment": parent_experiment,
        "parent_commit": parent_commit,
        "commit": commit,
        "objective_arm": objective_arm,
        "val_bpb": 0.0,
        "tok_per_sec": 0.0,
        "memory_gb": 0.0,
        "num_params_m": 0.0,
        "flops_per_token_g": 0.0,
        "mfu_percent": 0.0,
        "startup_seconds": 0.0,
        "training_seconds": 0.0,
        "total_seconds": 0.0,
        "quality_score": 0.0,
        "speed_score": 0.0,
        "memory_score": 0.0,
        "params_score": 0.0,
        "cost_score": 0.0,
        "global_reward": 0.0,
        "arm_reward": 0.0,
        "status": "crash",
        "description": description,
    }


def choose_next_objective(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"objective_arm": OBJECTIVE_ARMS[0], "reason": "cold start: no prior experiments"}

    stats = defaultdict(lambda: {"n": 0, "reward_sum": 0.0, "last_seen": -10**9})
    for idx, row in enumerate(rows, start=1):
        arm = row["objective_arm"]
        stats[arm]["n"] += 1
        stats[arm]["reward_sum"] += row["arm_reward"]
        stats[arm]["last_seen"] = idx

    # Cold-start coverage over objective arms.
    for arm in OBJECTIVE_ARMS:
        if stats[arm]["n"] == 0:
            return {"objective_arm": arm, "reason": f"cold start: arm '{arm}' has not been sampled yet"}

    t = max(1, len(rows))
    # Periodically force the least recently used arm so no objective starves.
    if (t + 1) % (len(OBJECTIVE_ARMS) + 3) == 0:
        least_recent = min(OBJECTIVE_ARMS, key=lambda arm: stats[arm]["last_seen"])
        return {
            "objective_arm": least_recent,
            "reason": f"forced exploration: '{least_recent}' has not been tried recently",
        }

    best_arm = OBJECTIVE_ARMS[0]
    best_score = -float("inf")
    scores: Dict[str, float] = {}
    for arm in OBJECTIVE_ARMS:
        n = stats[arm]["n"]
        mean_reward = stats[arm]["reward_sum"] / max(n, 1)
        score = mean_reward + UCB_EXPLORATION * math.sqrt(math.log(t + 1) / n)
        scores[arm] = score
        if score > best_score:
            best_score = score
            best_arm = arm

    return {
        "objective_arm": best_arm,
        "reason": "ucb1",
        "scores": scores,
    }


def pick_parent(rows: List[Dict[str, Any]], objective_arm: str) -> Dict[str, Any]:
    successes = [row for row in rows if is_success(row)]
    if not successes:
        return {
            "parent_experiment": 0,
            "parent_commit": "",
            "parent_reason": "no prior successful run; use the current branch tip as baseline",
        }

    frontier = frontier_rows(rows)
    candidates = frontier if frontier else successes

    if objective_arm == "quality":
        key_fn = lambda row: (row["val_bpb"], -row["global_reward"], row["experiment"])
    elif objective_arm == "speed":
        key_fn = lambda row: (-row["tok_per_sec"], row["val_bpb"], row["experiment"])
    elif objective_arm == "memory":
        key_fn = lambda row: (row["memory_gb"], row["val_bpb"], row["experiment"])
    elif objective_arm == "params":
        key_fn = lambda row: (row["num_params_m"], row["val_bpb"], row["experiment"])
    elif objective_arm == "cost":
        key_fn = lambda row: (row["flops_per_token_g"], row["val_bpb"], row["experiment"])
    else:
        key_fn = lambda row: (-row["global_reward"], row["experiment"])

    parent = min(candidates, key=key_fn)
    return {
        "parent_experiment": parent["experiment"],
        "parent_commit": parent["commit"],
        "parent_reason": f"objective anchor chosen from current Pareto frontier for '{objective_arm}'",
    }


def next_plan(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    objective = choose_next_objective(rows)
    objective_arm = objective["objective_arm"]
    parent = pick_parent(rows, objective_arm)

    plan = {
        "objective_arm": objective_arm,
        "objective_reason": objective["reason"],
        "parent_commit": parent["parent_commit"],
        "parent_experiment": parent["parent_experiment"],
        "parent_reason": parent["parent_reason"],
        "num_experiments": len(rows),
        "frontier_size": len(frontier_rows(rows)) if rows else 0,
    }
    if "scores" in objective:
        plan["ucb_scores"] = objective["scores"]
    return plan


def format_row_for_write(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": row["experiment"],
        "parent_experiment": row["parent_experiment"],
        "parent_commit": row["parent_commit"],
        "commit": row["commit"],
        "objective_arm": row["objective_arm"],
        "val_bpb": f"{row['val_bpb']:.6f}",
        "tok_per_sec": f"{row['tok_per_sec']:.1f}",
        "memory_gb": f"{row['memory_gb']:.3f}",
        "num_params_m": f"{row['num_params_m']:.3f}",
        "flops_per_token_g": f"{row['flops_per_token_g']:.3f}",
        "mfu_percent": f"{row['mfu_percent']:.2f}",
        "startup_seconds": f"{row['startup_seconds']:.2f}",
        "training_seconds": f"{row['training_seconds']:.2f}",
        "total_seconds": f"{row['total_seconds']:.2f}",
        "quality_score": f"{row['quality_score']:.4f}",
        "speed_score": f"{row['speed_score']:.4f}",
        "memory_score": f"{row['memory_score']:.4f}",
        "params_score": f"{row['params_score']:.4f}",
        "cost_score": f"{row['cost_score']:.4f}",
        "global_reward": f"{row['global_reward']:.4f}",
        "arm_reward": f"{row['arm_reward']:.4f}",
        "status": row['status'],
        "description": row['description'],
    }


def cmd_init(args: argparse.Namespace) -> None:
    write_header(args.results)
    print(f"Initialized {args.results}")


def cmd_next_plan(args: argparse.Namespace) -> None:
    rows = read_rows(args.results)
    plan = next_plan(rows)
    print(json.dumps(plan, indent=2, sort_keys=True))


def cmd_next_arm(args: argparse.Namespace) -> None:
    rows = read_rows(args.results)
    plan = next_plan(rows)
    print(plan["objective_arm"])


def cmd_frontier(args: argparse.Namespace) -> None:
    rows = read_rows(args.results)
    frontier = frontier_rows(rows)
    frontier.sort(key=lambda row: row["experiment"])
    print(json.dumps(frontier, indent=2, sort_keys=True))


def cmd_append(args: argparse.Namespace) -> None:
    rows = read_rows(args.results)
    experiment = len(rows) + 1
    summary = parse_summary_line(args.log)

    objective_arm = args.objective_arm
    if objective_arm not in OBJECTIVE_ARMS:
        raise SystemExit(f"Unknown objective arm: {objective_arm}. Expected one of {OBJECTIVE_ARMS}")

    if summary is None:
        row = empty_crash_row(
            experiment=experiment,
            commit=args.commit,
            parent_commit=args.parent_commit,
            parent_experiment=args.parent_experiment,
            objective_arm=objective_arm,
            description=args.description,
        )
    else:
        row = build_row_from_summary(
            summary=summary,
            experiment=experiment,
            commit=args.commit,
            parent_commit=args.parent_commit,
            parent_experiment=args.parent_experiment,
            default_objective_arm=objective_arm,
            description=args.description,
            existing_rows=rows,
        )

    append_row(args.results, format_row_for_write(row))
    print(json.dumps(row, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Objective-arm bandit controller helpers")
    sub = parser.add_subparsers(required=True)

    p_init = sub.add_parser("init", help="Initialize results.tsv")
    p_init.add_argument("--results", type=Path, required=True)
    p_init.set_defaults(func=cmd_init)

    p_next_plan = sub.add_parser("next-plan", help="Choose the next objective arm and parent commit")
    p_next_plan.add_argument("--results", type=Path, required=True)
    p_next_plan.set_defaults(func=cmd_next_plan)

    p_next_arm = sub.add_parser("next-arm", help="Print only the next objective arm")
    p_next_arm.add_argument("--results", type=Path, required=True)
    p_next_arm.set_defaults(func=cmd_next_arm)

    p_append = sub.add_parser("append", help="Append a completed run from run.log")
    p_append.add_argument("--results", type=Path, required=True)
    p_append.add_argument("--log", type=Path, required=True)
    p_append.add_argument("--commit", type=str, required=True)
    p_append.add_argument("--parent-commit", type=str, default="")
    p_append.add_argument("--parent-experiment", type=int, default=0)
    p_append.add_argument("--objective-arm", type=str, required=True)
    p_append.add_argument("--description", type=str, required=True)
    p_append.set_defaults(func=cmd_append)

    p_frontier = sub.add_parser("frontier", help="Print the current Pareto frontier")
    p_frontier.add_argument("--results", type=Path, required=True)
    p_frontier.set_defaults(func=cmd_frontier)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
