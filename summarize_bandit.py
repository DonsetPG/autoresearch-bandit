"""
Generate a static progress figure and an animated GIF for objective-arm bandit autoresearch.

Expected TSV columns are the ones produced by bandit_controller.py. The script can also
consume older bandit TSVs because it recomputes scores and frontier membership from the
raw metrics.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

from bandit_controller import OBJECTIVE_ARMS, frontier_rows, pareto_indices, read_rows


SCORE_FIELDS = [
    ("quality_score", "Quality"),
    ("speed_score", "Speed"),
    ("memory_score", "Memory"),
    ("params_score", "Params"),
    ("cost_score", "Cost"),
]


def running_best(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    best = -float("inf")
    for value in values:
        best = max(best, value)
        out.append(best)
    return out


def arm_order(rows: List[Dict]) -> List[str]:
    seen: List[str] = []
    for arm in OBJECTIVE_ARMS:
        if any(row["objective_arm"] == arm for row in rows):
            seen.append(arm)
    for row in rows:
        arm = row["objective_arm"]
        if arm not in seen:
            seen.append(arm)
    return seen


def arm_color_map(rows: List[Dict]) -> Dict[str, str]:
    arms = arm_order(rows)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {arm: cycle[i % len(cycle)] for i, arm in enumerate(arms)}


def arm_stats(rows: List[Dict]) -> List[Dict]:
    stats = defaultdict(lambda: {"pulls": 0, "keeps": 0, "reward_sum": 0.0})
    for row in rows:
        arm = row["objective_arm"]
        stats[arm]["pulls"] += 1
        stats[arm]["reward_sum"] += row["arm_reward"]
        if row["status"] == "keep":
            stats[arm]["keeps"] += 1
    out: List[Dict] = []
    for arm in arm_order(rows):
        pulls = stats[arm]["pulls"]
        out.append(
            {
                "objective_arm": arm,
                "pulls": pulls,
                "keeps": stats[arm]["keeps"],
                "mean_arm_reward": stats[arm]["reward_sum"] / pulls if pulls else 0.0,
            }
        )
    return out


def arm_heatmap(rows: List[Dict], arms: List[str]) -> List[List[float]]:
    matrix = [[math.nan for _ in rows] for _ in arms]
    for col, row in enumerate(rows):
        matrix[arms.index(row["objective_arm"])][col] = row["arm_reward"]
    return matrix


def running_best_scores(rows: List[Dict]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for field, _ in SCORE_FIELDS:
        out[field] = running_best([row[field] for row in rows])
    return out


def best_extremes(rows: List[Dict]) -> List[tuple[str, Dict]]:
    current_frontier = frontier_rows(rows)
    if not current_frontier:
        current_frontier = [row for row in rows if row["status"] != "crash"]
    if not current_frontier:
        return []
    return [
        ("Best quality", min(current_frontier, key=lambda r: (r["val_bpb"], -r["global_reward"]))),
        ("Fastest", max(current_frontier, key=lambda r: (r["tok_per_sec"], -r["val_bpb"]))),
        ("Lowest VRAM", min(current_frontier, key=lambda r: (r["memory_gb"], r["val_bpb"]))),
        ("Smallest", min(current_frontier, key=lambda r: (r["num_params_m"], r["val_bpb"]))),
        ("Lowest FLOPs/token", min(current_frontier, key=lambda r: (r["flops_per_token_g"], r["val_bpb"]))),
        ("Highest utility", max(current_frontier, key=lambda r: (r["global_reward"], -r["val_bpb"]))),
    ]


def draw_figure(rows: List[Dict], title: str, png_path: Path | None = None, dpi: int = 160):
    if not rows:
        raise ValueError("No rows to plot.")

    colors = arm_color_map(rows)
    arms = arm_order(rows)
    stats = arm_stats(rows)
    frontier_idx = set(pareto_indices(rows))
    experiments = [row["experiment"] for row in rows]
    global_rewards = [row["global_reward"] for row in rows]
    best_global_rewards = running_best(global_rewards)
    best_scores = running_best_scores(rows)

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0])

    # Panel 1: global utility through time.
    ax1 = fig.add_subplot(gs[0, 0])
    for i, row in enumerate(rows):
        if row["status"] == "crash":
            marker = "x"
            size = 44
        elif row["status"] == "keep":
            marker = "o"
            size = 72
        else:
            marker = "."
            size = 34
        edge = "black" if i in frontier_idx else "none"
        scatter_kwargs = {
            "s": size,
            "marker": marker,
            "c": [colors[row["objective_arm"]]],
            "alpha": 0.9,
        }
        if marker != "x":
            scatter_kwargs["edgecolors"] = edge
        ax1.scatter(
            row["experiment"],
            row["global_reward"],
            **scatter_kwargs,
        )
    ax1.plot(experiments, best_global_rewards, linewidth=2.0, label="Running best")
    ax1.set_title("Global multi-objective utility")
    ax1.set_xlabel("Experiment #")
    ax1.set_ylabel("Global reward")
    ax1.grid(alpha=0.25)

    # Panel 2: quality-speed Pareto view.
    ax2 = fig.add_subplot(gs[0, 1])
    successful_rows = [row for row in rows if row["status"] != "crash"]
    for row in successful_rows:
        idx = row["experiment"] - 1
        edge = "black" if idx in frontier_idx else "none"
        size = max(30.0, row["memory_gb"] * 14.0)
        ax2.scatter(
            row["tok_per_sec"],
            row["val_bpb"],
            s=size,
            c=[colors[row["objective_arm"]]],
            edgecolors=edge,
            alpha=0.85,
        )
    frontier = frontier_rows(rows)
    frontier = sorted(frontier, key=lambda r: r["tok_per_sec"])
    if frontier:
        ax2.plot([r["tok_per_sec"] for r in frontier], [r["val_bpb"] for r in frontier], linewidth=1.6, alpha=0.8)
    ax2.set_title("Quality vs speed (marker size = VRAM)")
    ax2.set_xlabel("Steady tokens / second")
    ax2.set_ylabel("Validation BPB (lower is better)")
    ax2.grid(alpha=0.25)

    # Panel 3: running best score per objective.
    ax3 = fig.add_subplot(gs[0, 2])
    for field, label in SCORE_FIELDS:
        ax3.plot(experiments, best_scores[field], linewidth=1.8, label=label)
    ax3.set_title("Running-best normalized objective scores")
    ax3.set_xlabel("Experiment #")
    ax3.set_ylabel("Score in [0, 1]")
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(alpha=0.25)
    ax3.legend(loc="lower right", fontsize=8)

    # Panel 4: arm allocation and mean arm reward.
    ax4 = fig.add_subplot(gs[1, 0])
    x = list(range(len(stats)))
    pulls = [s["pulls"] for s in stats]
    keeps = [s["keeps"] for s in stats]
    mean_rewards = [s["mean_arm_reward"] for s in stats]
    ax4.bar(x, pulls, label="Pulls", alpha=0.55)
    ax4.bar(x, keeps, label="Kept", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels([s["objective_arm"] for s in stats], rotation=20, ha="right")
    ax4.set_ylabel("Experiments")
    ax4.set_title("Objective-arm allocation")
    ax4.grid(axis="y", alpha=0.25)
    ax4b = ax4.twinx()
    ax4b.plot(x, mean_rewards, marker="o", linewidth=1.8, label="Mean arm reward")
    ax4b.set_ylabel("Mean arm reward")
    lines_1, labels_1 = ax4.get_legend_handles_labels()
    lines_2, labels_2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    # Panel 5: frontier summary card.
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    lines = ["Current frontier anchors"]
    for label, row in best_extremes(rows):
        lines.append(
            f"{label}: exp {row['experiment']} | arm={row['objective_arm']} | "
            f"bpb={row['val_bpb']:.4f}, tok/s={row['tok_per_sec']:.0f}, "
            f"VRAM={row['memory_gb']:.2f} GB, params={row['num_params_m']:.1f} M, "
            f"flops/tok={row['flops_per_token_g']:.2f} G"
        )
    text = "\n\n".join(lines)
    ax5.text(0.0, 1.0, text, va="top", ha="left", fontsize=10, family="monospace")

    # Panel 6: arm pull timeline heatmap.
    ax6 = fig.add_subplot(gs[1, 2])
    heat = arm_heatmap(rows, arms)
    masked = [[-1.0 if math.isnan(v) else v for v in row] for row in heat]
    img = ax6.imshow(masked, aspect="auto", interpolation="nearest", vmin=-1.0, vmax=1.0)
    ax6.set_title("Objective-arm pull timeline")
    ax6.set_xlabel("Experiment #")
    ax6.set_ylabel("Objective arm")
    ax6.set_yticks(range(len(arms)))
    ax6.set_yticklabels(arms)
    xticks = experiments[::max(1, len(experiments) // 8)]
    ax6.set_xticks([x - 1 for x in xticks])
    ax6.set_xticklabels(xticks)
    cbar = fig.colorbar(img, ax=ax6, fraction=0.046, pad=0.04)
    cbar.set_label("Arm reward (-1 = not pulled)")

    kept = sum(1 for row in rows if row["status"] == "keep")
    crashed = sum(1 for row in rows if row["status"] == "crash")
    current_frontier_size = len(frontier_rows(rows))
    subtitle = (
        f"{title}\n"
        f"{len(rows)} experiments, {kept} kept, {crashed} crashes, current frontier size {current_frontier_size}, "
        f"best global reward {max(best_global_rewards):.3f}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize objective-arm bandit autoresearch runs into a PNG and GIF.")
    parser.add_argument("results_tsv", type=Path, help="Path to results.tsv")
    parser.add_argument("--png", type=Path, default=Path("bandit_progress.png"), help="Output PNG path")
    parser.add_argument("--gif", type=Path, default=Path("bandit_progress.gif"), help="Output GIF path")
    parser.add_argument("--title", type=str, default="Autoresearch objective-bandit progress", help="Figure title")
    parser.add_argument("--skip-gif", action="store_true", help="Only create the PNG")
    args = parser.parse_args()

    rows = read_rows(args.results_tsv)
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
