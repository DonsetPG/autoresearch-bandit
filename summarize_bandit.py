
"""
Generate a static progress figure and an animated GIF for multi-objective bandit autoresearch runs.

Expected TSV columns:
experiment	commit	arm	val_bpb	memory_gb	tok_per_sec	mfu_percent	num_params_m	flops_per_token_g	reward	status	description

The `reward` column is optional; when missing it is recomputed from baseline-anchored scores.
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image


OBJECTIVE_WEIGHTS = {
    "quality": 0.45,
    "speed": 0.20,
    "memory": 0.15,
    "cost": 0.10,
    "stability": 0.10,
}

QUALITY_BAND = 0.010
RELATIVE_BAND = 0.25


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def parse_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def load_results(path: Path) -> List[Dict]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            rows.append(
                {
                    "experiment": int(row.get("experiment", len(rows) + 1)),
                    "commit": row.get("commit", ""),
                    "arm": row.get("arm", "unknown"),
                    "val_bpb": parse_float(row.get("val_bpb", "0"), default=0.0),
                    "memory_gb": parse_float(row.get("memory_gb", "0"), default=0.0),
                    "tok_per_sec": parse_float(row.get("tok_per_sec", "0"), default=0.0),
                    "mfu_percent": parse_float(row.get("mfu_percent", "0"), default=0.0),
                    "num_params_m": parse_float(row.get("num_params_m", "0"), default=0.0),
                    "flops_per_token_g": parse_float(row.get("flops_per_token_g", "0"), default=0.0),
                    "reward": row.get("reward", ""),
                    "status": row.get("status", "discard"),
                    "description": row.get("description", ""),
                }
            )
    return rows


def first_success(rows: List[Dict]) -> Dict:
    for row in rows:
        if row["status"] != "crash":
            return row
    raise ValueError("Need at least one non-crash run to anchor normalization.")


def compute_reward(row: Dict, baseline: Dict) -> Dict[str, float]:
    stability = 0.0 if row["status"] == "crash" else 1.0
    if stability == 0.0:
        return {
            "quality": 0.0,
            "speed": 0.0,
            "memory": 0.0,
            "cost": 0.0,
            "stability": 0.0,
            "reward": 0.0,
        }

    base_quality = baseline["val_bpb"]
    base_speed = max(baseline["tok_per_sec"], 1e-9)
    base_memory = max(baseline["memory_gb"], 1e-9)
    base_cost = max(baseline["flops_per_token_g"], 1e-9)

    quality = clamp(0.5 + (base_quality - row["val_bpb"]) / QUALITY_BAND)
    speed = clamp(0.5 + (row["tok_per_sec"] - base_speed) / (RELATIVE_BAND * base_speed))
    memory = clamp(0.5 + (base_memory - row["memory_gb"]) / (RELATIVE_BAND * base_memory))
    cost = clamp(0.5 + (base_cost - row["flops_per_token_g"]) / (RELATIVE_BAND * base_cost))

    reward = (
        OBJECTIVE_WEIGHTS["quality"] * quality
        + OBJECTIVE_WEIGHTS["speed"] * speed
        + OBJECTIVE_WEIGHTS["memory"] * memory
        + OBJECTIVE_WEIGHTS["cost"] * cost
        + OBJECTIVE_WEIGHTS["stability"] * stability
    )
    return {
        "quality": quality,
        "speed": speed,
        "memory": memory,
        "cost": cost,
        "stability": stability,
        "reward": reward,
    }


def enrich_rows(rows: List[Dict]) -> List[Dict]:
    baseline = first_success(rows)
    out = []
    for row in rows:
        row = dict(row)
        comp = compute_reward(row, baseline)
        row.update({
            "quality_score": comp["quality"],
            "speed_score": comp["speed"],
            "memory_score": comp["memory"],
            "cost_score": comp["cost"],
            "stability_score": comp["stability"],
        })
        if row["reward"] == "":
            row["reward"] = comp["reward"]
        else:
            row["reward"] = float(row["reward"])
        out.append(row)
    return out


def dominates(a: Dict, b: Dict) -> bool:
    if a["status"] == "crash" or b["status"] == "crash":
        return False
    no_worse = (
        a["val_bpb"] <= b["val_bpb"]
        and a["tok_per_sec"] >= b["tok_per_sec"]
        and a["memory_gb"] <= b["memory_gb"]
        and a["flops_per_token_g"] <= b["flops_per_token_g"]
    )
    strictly_better = (
        a["val_bpb"] < b["val_bpb"]
        or a["tok_per_sec"] > b["tok_per_sec"]
        or a["memory_gb"] < b["memory_gb"]
        or a["flops_per_token_g"] < b["flops_per_token_g"]
    )
    return no_worse and strictly_better


def pareto_indices(rows: List[Dict]) -> List[int]:
    good = [i for i, row in enumerate(rows) if row["status"] != "crash"]
    frontier = []
    for i in good:
        dominated = False
        for j in good:
            if i == j:
                continue
            if dominates(rows[j], rows[i]):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def running_best(values: List[float]) -> List[float]:
    out = []
    best = -float("inf")
    for value in values:
        best = max(best, value)
        out.append(best)
    return out


def arm_order(rows: List[Dict]) -> List[str]:
    seen = []
    for row in rows:
        arm = row["arm"]
        if arm not in seen:
            seen.append(arm)
    return seen


def arm_stats(rows: List[Dict]) -> List[Dict]:
    stats = defaultdict(lambda: {"pulls": 0, "keeps": 0, "reward_sum": 0.0})
    for row in rows:
        arm = row["arm"]
        stats[arm]["pulls"] += 1
        stats[arm]["reward_sum"] += row["reward"]
        if row["status"] == "keep":
            stats[arm]["keeps"] += 1
    ordered = []
    for arm in arm_order(rows):
        pulls = stats[arm]["pulls"]
        ordered.append(
            {
                "arm": arm,
                "pulls": pulls,
                "keeps": stats[arm]["keeps"],
                "mean_reward": stats[arm]["reward_sum"] / pulls if pulls else 0.0,
            }
        )
    return ordered


def arm_heatmap(rows: List[Dict], arms: List[str]):
    matrix = [[math.nan for _ in rows] for _ in arms]
    for col, row in enumerate(rows):
        matrix[arms.index(row["arm"])][col] = row["reward"]
    return matrix


def arm_color_map(rows: List[Dict]):
    arms = arm_order(rows)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {arm: cycle[i % len(cycle)] for i, arm in enumerate(arms)}


def draw_figure(rows: List[Dict], title: str, png_path: Path | None = None, dpi: int = 160):
    if not rows:
        raise ValueError("No rows to plot.")
    rows = enrich_rows(rows)
    frontier = pareto_indices(rows)
    colors = arm_color_map(rows)
    arms = arm_order(rows)
    stats = arm_stats(rows)

    experiments = [row["experiment"] for row in rows]
    rewards = [row["reward"] for row in rows]
    best_rewards = running_best(rewards)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    for row in rows:
        marker = "o" if row["status"] == "keep" else ("x" if row["status"] == "crash" else ".")
        size = 80 if row["status"] == "keep" else 32
        edge = "black" if row["experiment"] - 1 in frontier else "none"
        ax1.scatter(
            row["experiment"],
            row["reward"],
            s=size,
            marker=marker,
            c=[colors[row["arm"]]],
            edgecolors=edge,
            alpha=0.9,
        )
    ax1.plot(experiments, best_rewards, linewidth=2.0, label="Running best utility")
    ax1.set_title("Bandit utility by experiment")
    ax1.set_xlabel("Experiment #")
    ax1.set_ylabel("Scalarized multi-objective reward")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    for row in rows:
        if row["status"] == "crash":
            continue
        edge = "black" if row["experiment"] - 1 in frontier else "none"
        size = max(30.0, row["memory_gb"] * 12.0)
        ax2.scatter(
            row["tok_per_sec"],
            row["val_bpb"],
            s=size,
            c=[colors[row["arm"]]],
            edgecolors=edge,
            alpha=0.85,
        )
    frontier_rows = [rows[i] for i in frontier]
    frontier_rows.sort(key=lambda r: r["tok_per_sec"])
    if frontier_rows:
        ax2.plot(
            [r["tok_per_sec"] for r in frontier_rows],
            [r["val_bpb"] for r in frontier_rows],
            linewidth=1.8,
            alpha=0.9,
            label="Pareto frontier",
        )
    ax2.set_title("Quality-speed Pareto view (size = memory GB)")
    ax2.set_xlabel("Steady tokens / second")
    ax2.set_ylabel("Validation BPB (lower is better)")
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    x = list(range(len(stats)))
    pulls = [s["pulls"] for s in stats]
    keeps = [s["keeps"] for s in stats]
    means = [s["mean_reward"] for s in stats]
    ax3.bar(x, pulls, label="Pulls", alpha=0.55)
    ax3.bar(x, keeps, label="Keeps", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([s["arm"] for s in stats], rotation=25, ha="right")
    ax3.set_title("Arm allocation and keep count")
    ax3.set_ylabel("Experiments")
    ax3.grid(axis="y", alpha=0.25)
    ax3b = ax3.twinx()
    ax3b.plot(x, means, marker="o", linewidth=1.8, label="Mean reward")
    ax3b.set_ylabel("Mean reward")
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax4 = fig.add_subplot(gs[1, 1])
    heat = arm_heatmap(rows, arms)
    masked = [[-1 if math.isnan(v) else v for v in row] for row in heat]
    img = ax4.imshow(masked, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    ax4.set_title("Arm pull timeline (cell value = reward)")
    ax4.set_xlabel("Experiment #")
    ax4.set_ylabel("Arm")
    ax4.set_yticks(range(len(arms)))
    ax4.set_yticklabels(arms)
    xticks = experiments[::max(1, len(experiments) // 8)]
    ax4.set_xticks([x - 1 for x in xticks])
    ax4.set_xticklabels(xticks)
    cbar = fig.colorbar(img, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Reward (-1 = not pulled)")

    kept = sum(1 for row in rows if row["status"] == "keep")
    crashed = sum(1 for row in rows if row["status"] == "crash")
    frontier_count = len(frontier)
    subtitle = (
        f"{title}\n"
        f"{len(rows)} experiments, {kept} kept, {crashed} crashes, "
        f"frontier size {frontier_count}, best reward {max(best_rewards):.3f}"
    )
    fig.suptitle(subtitle, fontsize=15)

    handles = []
    labels = []
    for arm in arms:
        handle = ax1.scatter([], [], c=[colors[arm]], label=arm)
        handles.append(handle)
        labels.append(arm)
    ax1.legend(handles, labels, loc="best", fontsize=8)

    if png_path is not None:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def save_gif(rows: List[Dict], title: str, gif_path: Path, dpi: int = 120, duration_ms: int = 350):
    rows = enrich_rows(rows)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        frames = []
        for end in range(1, len(rows) + 1):
            frame_path = tmp / f"frame_{end:04d}.png"
            draw_figure(rows[:end], title=title, png_path=frame_path, dpi=dpi)
            frames.append(Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE))
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        for frame in frames:
            frame.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize bandit autoresearch runs into a PNG and GIF.")
    parser.add_argument("results_tsv", type=Path, help="Path to results.tsv")
    parser.add_argument("--png", type=Path, default=Path("bandit_progress.png"), help="Output PNG path")
    parser.add_argument("--gif", type=Path, default=Path("bandit_progress.gif"), help="Output GIF path")
    parser.add_argument("--title", type=str, default="Autoresearch bandit progress", help="Figure title")
    parser.add_argument("--skip-gif", action="store_true", help="Only create the PNG")
    args = parser.parse_args()

    rows = load_results(args.results_tsv)
    if not rows:
        raise SystemExit("results.tsv is empty.")

    draw_figure(rows, title=args.title, png_path=args.png)
    if not args.skip_gif:
        save_gif(rows, title=args.title, gif_path=args.gif)

    print(f"Wrote {args.png}")
    if not args.skip_gif:
        print(f"Wrote {args.gif}")


if __name__ == "__main__":
    main()
