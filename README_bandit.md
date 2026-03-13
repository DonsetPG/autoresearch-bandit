# autoresearch-bandit

This is a multi-objective, bandit-driven variant of autoresearch.

The original repo is a single-objective outer loop: modify `train.py`, run for a fixed 5-minute budget, and keep the commit only when `val_bpb` gets lower. That design is clean and effective, but it collapses every tradeoff into one number. This variant keeps `val_bpb` as the primary quality metric while also optimizing speed, memory, inference cost, and stability at the orchestration layer.

## What stays fixed

`prepare.py` is still read-only. It still defines the fixed data pipeline, tokenizer, time budget, and `evaluate_bpb` function. The training loss inside a single run is still standard next-token cross-entropy. The multi-objective logic lives outside the training step, in the agent loop that decides which experiment family to try next.

## What changes

This variant introduces four new pieces:

- **`train_bandit.py`** — the training script, based on the original `train.py`, but with richer end-of-run telemetry and a machine-readable `run_summary_json` line.
- **`program_bandit.md`** — agent instructions for multi-objective search with a discrete multi-arm bandit over experiment families.
- **`bandit_controller.py`** — helper CLI for `init`, `append`, `next-arm`, and `frontier`.
- **`results.tsv`** — expanded run log with raw metrics and a scalarized reward.
- **`summarize_bandit.py`** — turns `results.tsv` into a static figure and an animated GIF.

## Objectives

By default the controller optimizes 5 goals:

1. `val_bpb` — lower is better.
2. `steady_tok_per_sec` — higher is better.
3. `peak_vram_gb` — lower is better.
4. `num_flops_per_token_G` — lower is better.
5. stability — successful, non-crashing runs are better.

`mfu_percent` and `num_params_M` are logged as diagnostics.

## Bandit design

Each experiment family is treated as one arm:

- `retest_frontier`
- `optimizer`
- `schedule`
- `shape`
- `attention_rope`
- `residual_ve`
- `mlp_block`
- `batch_efficiency`

The outer loop uses a simple UCB policy over arm families. Reward is a weighted scalarization of the 5 objectives, with quality dominant. Keep / discard is based on Pareto dominance, not only on `val_bpb`.

This gives two decision layers:

- **bandit allocation** decides which subsystem to explore next
- **Pareto retention** decides whether the new commit should remain on the branch

## Results schema

`results.tsv` uses this header:

```tsv
experiment	commit	arm	val_bpb	memory_gb	tok_per_sec	mfu_percent	num_params_m	flops_per_token_g	reward	status	description
```

`status` is one of:

- `keep`
- `discard`
- `crash`

## Figure design

`summarize_bandit.py` produces a 2x2 summary figure:

- **top-left:** scalarized bandit reward by experiment, plus running best utility
- **top-right:** Pareto view of quality vs speed, with marker size proportional to memory
- **bottom-left:** arm pulls, keep counts, and mean reward by arm
- **bottom-right:** arm-pull timeline heatmap, where each cell shows reward for the selected arm on that experiment

It also produces a GIF that animates the same figure experiment by experiment.

## Quick start

```bash
# 1. Run a single experiment
AUTORESEARCH_RUN_TAG=mar13-bandit \
AUTORESEARCH_RUN_ID=1 \
AUTORESEARCH_ARM=optimizer \
AUTORESEARCH_NOTE="baseline" \
uv run train_bandit.py > run.log 2>&1

# 2. Append the result to results.tsv
python bandit_controller.py append --results results.tsv --log run.log --commit "$(git rev-parse --short HEAD)" --arm optimizer --description "baseline"

# 3. Choose the next arm
python bandit_controller.py next-arm --results results.tsv

# 4. Generate summaries
python summarize_bandit.py results.tsv --png bandit_progress.png --gif bandit_progress.gif
```

## Recommended workflow

1. Keep `prepare.py` unchanged.
2. Let the agent operate mainly on `train_bandit.py`.
3. Log every run to `results.tsv`.
4. Recompute the next arm from the logged rewards.
5. Regenerate the figure and GIF after each experiment or every few experiments.

## Why this is a better fit for multi-objective search

A pure “best loss wins” loop is brittle when the repo must balance hardware cost and quality. A multi-objective bandit gives you:

- explicit control over tradeoffs
- exploration pressure across research themes
- a retained Pareto frontier instead of a single incumbent
- a visual summary that reflects the real search space

If you later want more objectives, the clean extensions are downstream eval, inference latency, and robustness across seeds.
