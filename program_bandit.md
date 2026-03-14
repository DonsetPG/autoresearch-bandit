# autoresearch-bandit

This is the multi-objective, multi-arm-bandit version of autoresearch.

In this version, a **bandit arm is an optimization objective**, not a code area. The agent is free to edit **any repo file it wants** in pursuit of the chosen objective, as long as it preserves the fixed evaluation contract described below.

## Fixed contract

These items are **not** fair game during a live research run because changing them would break comparability across experiments:

- the dataset and data split
- tokenizer training and tokenizer directory layout
- `evaluate_bpb` and the definition of `val_bpb`
- the fixed wall-clock training budget used for comparison
- the results schema in `results.tsv`
- the reward equations in `bandit_controller.py`
- the summary logic in `summarize_bandit.py`

Everything else is editable. In particular, the agent may modify:

- `train_bandit.py`
- any helper modules it creates
- optimizer code
- model architecture code
- scheduling code
- kernels / utility wrappers
- logging helpers, as long as the final `run_summary_json` still exists and still contains the required fields

Never commit these runtime artifacts: `results.tsv`, `run.log`, `bandit_progress.png`, `bandit_progress.gif`. Leave them untracked.

## Candidate objectives for LLM autoresearch

Classically, there are many things one could optimize for a language model run:

- validation loss / BPB / perplexity
- training throughput
- peak VRAM
- parameter count / checkpoint size
- inference FLOPs per token
- end-to-end latency
- compile / startup overhead
- long-context quality
- benchmark score on downstream tasks
- sample quality / human preference
- robustness / crash rate / NaN resistance
- data efficiency / time-to-target

For this repo, use exactly **5 implemented objective arms** because they are measurable on every run and already available from the training summary:

1. `quality`  → improve `val_bpb` (lower is better)
2. `speed`    → improve `steady_tok_per_sec` (higher is better)
3. `memory`   → reduce `peak_vram_gb`
4. `params`   → reduce `num_params_M`
5. `cost`     → reduce `num_flops_per_token_G`

Crash-free execution is a **hard guardrail**. A crash gets zero reward and is never kept, but crash-robustness is not its own arm.

## High-level strategy

There are two separate decisions:

1. **Bandit allocation**: which objective arm should the next experiment focus on?
2. **Archive decision**: should the resulting commit be kept on the multi-objective Pareto frontier?

The bandit chooses the objective arm. The keep/discard decision is based on Pareto dominance across the 5 raw metrics.

## Parent selection and frontier archive

Because there is no single scalar "best commit" anymore, do **not** treat the current branch tip as the only parent.

Instead:

- maintain an archive of frontier commits
- before each experiment, choose a **parent commit** from the current frontier
- after a successful keep, create a lightweight git tag so that commit remains reachable

Use tags like:

```bash
git tag -f frontier-exp-0012 <commit>
```

This matters because later experiments may branch from different frontier points.

## Setup

1. Propose a fresh tag, for example `mar14-objective-bandit`.
2. Create a branch such as `autoresearch-bandit/<tag>`.
3. Read these files carefully:
   - `README_bandit.md`
   - `prepare.py`
   - `train_bandit.py`
   - `bandit_controller.py`
   - `summarize_bandit.py`
   - `program_bandit.md`
4. Verify that `~/.cache/autoresearch/` already contains the dataset shards and tokenizer. If not, ask the human to run:

```bash
uv run prepare.py
```

5. Initialize the results file:

```bash
python bandit_controller.py init --results results.tsv
```

6. Run a baseline experiment with objective arm `quality` and no parent commit override if the file is empty.
7. After that, continue autonomously. Do not pause for more permission.

## Reward model

The first successful run is the baseline.

Convert each raw metric to a normalized score in `[0, 1]`:

- `quality_score = clamp(0.5 + (baseline_val_bpb - val_bpb) / 0.010)`
- `speed_score   = clamp(0.5 + (tok_per_sec - baseline_tok_per_sec) / (0.25 * baseline_tok_per_sec))`
- `memory_score  = clamp(0.5 + (baseline_memory_gb - memory_gb) / (0.25 * baseline_memory_gb))`
- `params_score  = clamp(0.5 + (baseline_params_m - params_m) / (0.25 * baseline_params_m))`
- `cost_score    = clamp(0.5 + (baseline_flops_g - flops_g) / (0.25 * baseline_flops_g))`

Then compute the global multi-objective reward:

```text
global_reward =
    0.40 * quality_score +
    0.20 * speed_score   +
    0.15 * memory_score  +
    0.10 * params_score  +
    0.15 * cost_score
```

The bandit update uses an **arm-specific reward** so the chosen objective really matters:

```text
arm_reward = 0.65 * selected_objective_score + 0.35 * global_reward
```

Examples:

- if the chosen arm is `speed`, then `selected_objective_score = speed_score`
- if the chosen arm is `memory`, then `selected_objective_score = memory_score`

Crashes receive `global_reward = 0` and `arm_reward = 0`.

## Keep / discard rule

A successful run is `keep` if it lies on the observed Pareto frontier over the 5 raw objectives:

- minimize `val_bpb`
- maximize `tok_per_sec`
- minimize `memory_gb`
- minimize `num_params_m`
- minimize `flops_per_token_g`

Otherwise it is `discard`.

A crash is `crash`.

The bandit reward decides what to explore next. The Pareto frontier decides what survives as research output.

## Bandit policy

Use UCB1 over the 5 objective arms.

Cold start:

- pull every objective arm once
- prefer the order `quality`, `speed`, `memory`, `params`, `cost`

After cold start:

```text
ucb[a] = mean_arm_reward[a] + 0.35 * sqrt(log(t + 1) / n[a])
```

where:

- `t` is the number of completed experiments
- `n[a]` is the number of pulls of arm `a`
- `mean_arm_reward[a]` is the average `arm_reward` seen when arm `a` was chosen

Also force a least-recently-used exploration step periodically so no objective starves.

## How to choose the parent commit

Use the helper:

```bash
python bandit_controller.py next-plan --results results.tsv
```

This returns JSON containing:

- `objective_arm`
- `parent_commit`
- `parent_experiment`
- a short reason for both choices

Interpretation:

- `quality` arm: usually branch from the current frontier point with the best `val_bpb`
- `speed` arm: usually branch from the fastest frontier point
- `memory` arm: usually branch from the lowest-VRAM frontier point
- `params` arm: usually branch from the smallest frontier point
- `cost` arm: usually branch from the lowest-FLOPs frontier point

## Running one experiment

1. Query the next plan with `next-plan`.
2. Reset the worktree to the suggested parent commit if one exists.
3. Make one coherent set of changes anywhere in the repo that are intended to improve the chosen objective arm.
4. Commit the change.
5. Run the experiment.
6. Append the result to `results.tsv`.
7. Keep or discard according to the logged status.
8. Regenerate the summary PNG and GIF.
9. Continue immediately.

### Important constraint on edits

A chosen objective arm does **not** constrain where you edit. It only constrains **why** you are editing.

Examples:

- if the chosen arm is `speed`, you may change model shape, optimizer logic, compilation behavior, attention kernels, or batch sizing — anything that plausibly increases throughput
- if the chosen arm is `memory`, you may reduce width, change KV sharing, alter activation storage, or adjust microbatching — anywhere in the repo
- if the chosen arm is `quality`, you may change architecture, optimizer, schedule, residual paths, or data-order details — anywhere in the repo

The objective arm is the hypothesis label, not the file scope.

## Canonical command sequence

### 1) Ask the controller for the next plan

```bash
python bandit_controller.py next-plan --results results.tsv
```

### 2) Reset to the suggested parent if present

If `parent_commit` is non-empty:

```bash
git reset --hard <parent_commit>
```

### 3) Make code changes and commit them

```bash
git add -A
# then remove any runtime artifacts from the index if needed
git restore --staged results.tsv run.log bandit_progress.png bandit_progress.gif 2>/dev/null || true
git commit -m "<objective arm>: <brief hypothesis>"
```

### 4) Run the experiment

```bash
AUTORESEARCH_RUN_TAG=<tag> \
AUTORESEARCH_RUN_ID=<n> \
AUTORESEARCH_OBJECTIVE_ARM=<objective_arm> \
AUTORESEARCH_PARENT_COMMIT=<parent_commit> \
AUTORESEARCH_PARENT_EXPERIMENT=<parent_experiment> \
AUTORESEARCH_NOTE="<short description>" \
uv run train_bandit.py > run.log 2>&1
```

### 5) Append the result

```bash
python bandit_controller.py append \
  --results results.tsv \
  --log run.log \
  --commit "$(git rev-parse --short HEAD)" \
  --parent-commit "<parent_commit>" \
  --parent-experiment <parent_experiment> \
  --objective-arm <objective_arm> \
  --description "<short description>"
```

### 6) Keep or revert

If the row status is `keep`:

```bash
git tag -f frontier-exp-<NNNN> "$(git rev-parse HEAD)"
```

If the row status is `discard` or `crash` and `parent_commit` is non-empty:

```bash
git reset --hard <parent_commit>
```

### 7) Regenerate summary artifacts

```bash
python summarize_bandit.py results.tsv \
  --png bandit_progress.png \
  --gif bandit_progress.gif \
  --title "Autoresearch objective-bandit progress"
```

## Interpretation rules

- Prefer small, attributable diffs.
- Do not make a kitchen-sink patch unless the repo is clearly bottlenecked by one broad issue.
- Quality still matters the most globally. Do not destroy `val_bpb` to gain a little speed.
- A keep is not "the one best model". It is one non-dominated point on the current tradeoff frontier.
- Re-testing a frontier point is allowed when you suspect noise, but the default workflow should keep moving.

## Deliverables that should always exist

At any time, the working directory should be able to produce:

- `results.tsv`
- `bandit_progress.png`
- `bandit_progress.gif`
- a set of `frontier-exp-*` git tags for kept frontier commits
- a current branch containing the latest experiment code

## Autonomy rule

After setup, do not stop on your own and do not ask the human whether to continue. Keep running the bandit loop until interrupted.
