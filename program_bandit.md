# autoresearch-bandit

This is the multi-objective, multi-arm-bandit version of autoresearch.

The original repo advances the branch only when `val_bpb` improves. In this version, `val_bpb` is still the primary quality metric, but the outer-loop controller also optimizes speed, memory, inference cost, and stability. `prepare.py` remains read-only and its `evaluate_bpb` function remains the ground-truth quality evaluator. The bandit lives at the experiment-orchestration layer, not inside the minibatch training loss.

## Objectives

Optimize these 5 goals together:

1. **Quality** — `val_bpb` from the fixed validation harness. Lower is better.
2. **Training speed** — `steady_tok_per_sec`. Higher is better.
3. **Memory** — `peak_vram_gb`. Lower is better.
4. **Inference / model cost** — `num_flops_per_token_G`. Lower is better.
5. **Operational stability** — crash-free runs and no exploding loss. Higher is better.

`mfu_percent` and `num_params_M` are useful diagnostics, but they are not part of the scalar reward by default.

## Arm catalog

Treat each of these as one bandit arm. A single experiment should usually pull exactly one arm family.

- `retest_frontier` — rerun a current frontier configuration or the best-known config to estimate noise.
- `optimizer` — `EMBEDDING_LR`, `UNEMBEDDING_LR`, `MATRIX_LR`, `SCALAR_LR`, `ADAM_BETAS`, `WEIGHT_DECAY`, Muon settings.
- `schedule` — `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`, momentum / decay schedules.
- `shape` — `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `n_kv_head`, width-depth tradeoffs.
- `attention_rope` — `WINDOW_PATTERN`, short/long context patterning, RoPE base or rotary choices, KV sharing.
- `residual_ve` — value embeddings, residual lambda init, x0 path, gating structure.
- `mlp_block` — activation choice, expansion ratio, gated MLP variants, projection init.
- `batch_efficiency` — `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, grad accumulation, compile-friendly shapes.

Do not make “kitchen sink” edits. Keep a run attributable to one arm whenever possible.

## Setup

1. Propose a fresh tag, e.g. `mar13-bandit`.
2. Create branch `autoresearch-bandit/<tag>` from current master.
3. Read these files for context:
   - `README_bandit.md`
   - `prepare.py` (read only)
   - `train_bandit.py`
   - `program_bandit.md`
   - `bandit_controller.py`
   - `summarize_bandit.py`
4. Verify `~/.cache/autoresearch/` exists and contains data shards + tokenizer. If not, tell the human to run `uv run prepare.py`.
5. Initialize `results.tsv` with:

```bash
python bandit_controller.py init --results results.tsv
```

6. Confirm setup, then begin. Do not wait for more permission after setup.

## Reward model

Use a baseline-anchored scalar reward so the bandit can compare arms while still respecting multiple goals.

Let the first successful run be the baseline. Convert raw metrics into scores in `[0, 1]`:

- `quality_score = clamp(0.5 + (baseline_val_bpb - val_bpb) / 0.010)`
- `speed_score   = clamp(0.5 + (tok_per_sec - baseline_tok_per_sec) / (0.25 * baseline_tok_per_sec))`
- `memory_score  = clamp(0.5 + (baseline_memory_gb - memory_gb) / (0.25 * baseline_memory_gb))`
- `cost_score    = clamp(0.5 + (baseline_flops_g - flops_per_token_g) / (0.25 * baseline_flops_g))`
- `stability_score = 1.0` for successful runs, `0.0` for crashes

Then compute:

```text
reward =
    0.45 * quality_score +
    0.20 * speed_score +
    0.15 * memory_score +
    0.10 * cost_score +
    0.10 * stability_score
```

This keeps quality dominant, but it no longer allows the search to ignore speed, memory, or cost.

## Keep / discard rule

Do not use “best `val_bpb` only”.

A successful run is `keep` if it is on the observed Pareto frontier over:

- minimize `val_bpb`
- maximize `tok_per_sec`
- minimize `memory_gb`
- minimize `flops_per_token_g`

Otherwise mark it `discard`.

Use `crash` only when the run fails or produces no final summary.

This means the branch can advance on different tradeoff fronts, not just a single scalar optimum. The scalar reward drives arm allocation; the Pareto frontier drives which commits are retained as meaningful outcomes.

## Bandit policy

Use UCB1 over arm families. It is simple, deterministic, and easy to recompute from `results.tsv`.

Cold start:
- Pull every arm once before relying on UCB.
- Reserve at least one early pull for `retest_frontier`.

After cold start:
- For arm `a`, let `mean_reward[a]` be the average reward of all completed pulls for that arm.
- Let `n[a]` be the number of pulls for that arm.
- Let `t` be the total number of completed experiments.
- Compute:

```text
ucb[a] = mean_reward[a] + 0.35 * sqrt(log(t + 1) / n[a])
```

Choose the arm with the highest `ucb[a]`.

Exploration hygiene:
- Every 8th experiment, force either `retest_frontier` or the least recently pulled arm.
- If an arm crashes twice in a row for the same idea family, cool it down for 5 experiments.
- If an arm has been pulled 5+ times with no keep and mean reward below baseline, downweight it unless it has a very different hypothesis.

## Running one experiment

1. Choose the next arm.
2. Make one coherent edit in `train_bandit.py` corresponding to that arm.
3. Commit the change.
4. Run:

```bash
AUTORESEARCH_RUN_TAG=<tag> \
AUTORESEARCH_RUN_ID=<n> \
AUTORESEARCH_ARM=<arm> \
AUTORESEARCH_NOTE="<short description>" \
uv run train_bandit.py > run.log 2>&1
```

5. If the run crashes, inspect:

```bash
tail -n 80 run.log
```

6. If it succeeded, parse the final JSON summary line and append one row to `results.tsv`.

## Logging helper

Append a run with the controller:

```bash
python bandit_controller.py append \
  --results results.tsv \
  --log run.log \
  --commit "$(git rev-parse --short HEAD)" \
  --arm <arm> \
  --description "<description>"
```

The controller will:
- parse `run_summary_json` from `run.log`
- compute the scalar reward
- decide `keep` vs `discard` using the observed Pareto frontier
- append the row to `results.tsv`

If no final summary exists, it records a `crash` row.

## Choosing the next arm from `results.tsv`

Use this helper after every logged experiment:

```bash
python bandit_controller.py next-arm --results results.tsv
```

## Experiment loop

Loop forever:

1. Compute the next arm with the helper above.
2. Propose one concrete hypothesis in that arm.
3. Edit `train_bandit.py`.
4. Commit.
5. Run the experiment and capture `run.log`.
6. Append the row to `results.tsv`.
7. If status is `keep`, leave the commit in place.
8. If status is `discard` or `crash`, reset back to the last kept commit.
9. Regenerate the summary artifacts:

```bash
python summarize_bandit.py results.tsv --png bandit_progress.png --gif bandit_progress.gif --title "Autoresearch bandit progress"
```

10. Continue without asking the human whether to proceed.

## Interpretation rules

- Quality remains the dominant objective. Do not accept a large `val_bpb` regression merely because the run was cheaper.
- `retest_frontier` is not wasted compute. It estimates noise and prevents the bandit from overcommitting to lucky wins.
- Simple changes are preferred when rewards are close.
- If two runs are both on the frontier, prefer the simpler diff and the more reproducible one.

## Final deliverables

At any moment, the repo should contain:

- `results.tsv` with all experiments
- `bandit_progress.png`
- `bandit_progress.gif`
- the current best frontier commit on the branch
- the current `train_bandit.py` and `program_bandit.md`

Never stop on your own. The loop only stops when the human interrupts you.
