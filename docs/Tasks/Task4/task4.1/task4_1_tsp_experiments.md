# Task 4.1 — TSP Experiments (TSP50, TSP100)

**PRD link:** `docs/PRD.md` → Phase 4 → T4.1  
**Deliverable:** results tables + training curves for TSP50 and TSP100  
**Execution target:** Google Colab (TPU preferred; GPU fallback)

---

## 0. What this task is proving

For TSP sizes 50 and 100, we want a clean, apples-to-apples comparison of:

- `pomo` (RL4CO baseline)
- `leader_reward` (heuristic baseline)
- `maxk_pomo` (principled Max@K estimator)

Reported artifacts:

- A results table for greedy and best-of-K sampling evaluation.
- Training curves (validation cost vs epoch).

---

## 1. The implementation entrypoint

Use the Phase 4 harness:

- `code/src/experiments/phase4_tsp_experiments.py`

It wraps the already-existing Phase 3 scripts:

- Training: `code/src/experiments/train_tsp.py`
- Evaluation: `code/src/experiments/evaluate.py`

---

## 2. Colab runbook (copy/paste)

### 2.1 Assumptions

- You have the repo available under `REPO_DIR`.
- You set `PYTHONPATH` so `src.*` imports work:
  - `export PYTHONPATH="$REPO_DIR/code"`
- You write outputs to Drive for persistence:
  - `RUNS_DIR="$DRIVE_ROOT/runs_t4_1"`
  - `ARTIFACTS_DIR="$DRIVE_ROOT/artifacts_t4_1"`

### 2.2 Train (TPU)

```bash
%%bash
set -euo pipefail

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/code"

python3 -m src.experiments.phase4_tsp_experiments train \
  --runs_dir "$RUNS_DIR" \
  --artifacts_dir "$ARTIFACTS_DIR" \
  --data_dir "$RUNS_DIR/data" \
  --num_locs 50 100 \
  --seeds 0 1 2 \
  --algorithms pomo leader_reward maxk_pomo \
  --num_starts 16 \
  --leader_alpha 0.5 \
  --maxk_k 8 \
  --maxk_variance_reduction subloo \
  --max_epochs 10 \
  --batch_size 128 \
  --train_data_size 20000 \
  --val_data_size 2000 \
  --test_data_size 2000 \
  --lr 1e-4 \
  --accelerator tpu \
  --devices 8 \
  --precision 32-true \
  --generate_default_data
```

GPU fallback: same command, but change:

- `--accelerator gpu --devices 1`
- optional: `--precision 16-mixed`

### 2.3 Evaluate (greedy + best-of-K)

```bash
%%bash
set -euo pipefail

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/code"

python3 -m src.experiments.phase4_tsp_experiments eval \
  --runs_dir "$RUNS_DIR" \
  --artifacts_dir "$ARTIFACTS_DIR" \
  --num_locs 50 100 \
  --seeds 0 1 2 \
  --algorithms pomo leader_reward maxk_pomo \
  --num_starts 16 \
  --leader_alpha 0.5 \
  --maxk_k 8 \
  --maxk_variance_reduction subloo \
  --methods greedy sampling \
  --k_eval 128 \
  --num_instances 1000 \
  --batch_size 256 \
  --device cpu \
  --eval_seed 1234
```

If you have a GPU runtime available, set `--device cuda` for faster evaluation.

### 2.4 Summarize (tables + plots)

```bash
%%bash
set -euo pipefail

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/code"

python3 -m src.experiments.phase4_tsp_experiments summarize \
  --runs_dir "$RUNS_DIR" \
  --artifacts_dir "$ARTIFACTS_DIR" \
  --num_locs 50 100 \
  --seeds 0 1 2 \
  --algorithms pomo leader_reward maxk_pomo \
  --num_starts 16 \
  --leader_alpha 0.5 \
  --maxk_k 8 \
  --maxk_variance_reduction subloo \
  --methods greedy sampling \
  --k_eval 128
```

Outputs:

- Results table (markdown): `ARTIFACTS_DIR/tables/t4_1_tsp_results.md`
- Raw eval rows (CSV): `ARTIFACTS_DIR/tables/t4_1_tsp_eval_rows.csv`
- Training curves (CSV): `ARTIFACTS_DIR/tables/t4_1_tsp_training_curves.csv`
- Training curve plots: `ARTIFACTS_DIR/plots/t4_1_tsp50_val_cost.png` and `...tsp100...`

---

## 3. Notes / knobs

- Keep `num_starts` fixed across algorithms to make the comparison fair.
- `leader_alpha`, `maxk_k`, and `maxk_variance_reduction` are *not* tuned here; tuning/ablations belong in T4.7/T4.8.
- If you change `k_eval`, rerun `eval` and `summarize` with the same `--k_eval`.

