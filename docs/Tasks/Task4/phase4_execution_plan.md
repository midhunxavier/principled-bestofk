# Task 4 — Phase 4 Execution Plan: Experiments & Validation (Week 7–10)

**Phase:** 4 (Experiments & Validation)  
**Scope:** T4.1–T4.8 from `docs/PRD.md`  
**Status:** Planned  
**Owner:** Research Team  
**Last updated:** 2026-01-19  

---

## 0. Purpose (what Phase 4 is trying to prove)

Phase 4 exists to validate that our **principled, unbiased Max@K policy-gradient estimator** is:

1. **Better aligned to evaluation** than risk-neutral REINFORCE/POMO.
2. **More stable (lower-variance)** than the heuristic Leader Reward baseline.
3. **Competitively performant** across tasks and sizes (TSP50/100, CVRP, OP, PCTSP).

This phase produces the paper-ready evidence:

- Training curves (cost vs. wall-clock, vs. steps/epochs).
- Best-of-K evaluation results (Max@K at inference).
- Gradient-quality diagnostics (variance proxies + ESS).
- Ablations over `k` and variance-reduction strategy (none / Sample-LOO / SubLOO).

PRD checklist mapping:

- T4.1–T4.4: task results tables + curves (`docs/PRD.md:280`)
- T4.5: variance plots (`docs/PRD.md:284`)
- T4.6: convergence comparisons (`docs/PRD.md:285`)
- T4.7: `k` ablation (`docs/PRD.md:286`)
- T4.8: LOO baseline ablation (`docs/PRD.md:287`)

---

## 1. What we will run (repo entrypoints)

Phase 3 already implemented the “experiment harness” scripts we will use in Phase 4:

- Training (TSP only today): `code/src/experiments/train_tsp.py`
- Evaluation (best-of-K / Max@K metrics): `code/src/experiments/evaluate.py`
- Diagnostics (gradient variance proxies + ESS): `code/src/experiments/diagnose_gradients.py`
- Phase 4 TSP runner (T4.1): `code/src/experiments/phase4_tsp_experiments.py`

Task-specific runbook:

- T4.1 (TSP50/100): `docs/Tasks/Task4/task4_1_tsp_experiments.md`

Algorithms supported by the scripts:

- `pomo` (RL4CO baseline)
- `leader_reward` (heuristic baseline)
- `maxk_pomo` (principled estimator)

Important definitions:

- `n`: number of trajectories per instance during training; in POMO-style training this is `n = num_starts`.
- `k`: the Max@K parameter for the training objective (must satisfy `1 <= k <= n`).

---

## 2. Colab TPU strategy (pragmatic, with a go/no-go smoke test)

### 2.1 Reality check: TPU support risk

Our stack is PyTorch + Lightning + RL4CO. Training on Colab TPUs requires `torch_xla` and that the full dependency chain uses ops supported by XLA.

**Plan:** start Phase 4 with a TPU smoke test on a tiny run. If it fails due to unsupported ops, we switch the same plan to Colab GPU (no other changes).

We still structure the run commands so they are copy/paste friendly in Colab and write outputs to Google Drive so that switching runtime (TPU ⇄ GPU) is painless.

### 2.2 Colab notebook cell template (TPU)

Copy/paste these cells into a new Colab notebook.

#### Cell A — Mount Drive + define paths

```python
import os
from google.colab import drive

drive.mount("/content/drive")

# Change these to wherever you want outputs / the repo to live
os.environ["DRIVE_ROOT"] = "/content/drive/MyDrive/principled_bestofk_phase4"
os.environ["REPO_DIR"] = "/content/principled-bestofk"
```

#### Cell B — Clone repo

```bash
%%bash
set -euo pipefail

rm -rf "$REPO_DIR"

# Option A (recommended): clone from GitHub
GIT_URL="https://github.com/<org-or-user>/principled-bestofk.git"
git clone "$GIT_URL" "$REPO_DIR"

# Option B (no GitHub): copy/unzip the repo from Drive into /content/
# - If you have a zip at "$DRIVE_ROOT/src/principled-bestofk.zip":
#   unzip -q "$DRIVE_ROOT/src/principled-bestofk.zip" -d /content
#   mv /content/principled-bestofk "$REPO_DIR"

cd "$REPO_DIR"
```

Notes:
- If you don’t use GitHub, simplest is to zip the repo, upload to Drive, then unzip into `/content/`.
- Keep the repo on `/content/` for speed; keep outputs on Drive for persistence.

#### Cell C — Install dependencies (TPU)

TPU requires `torch_xla`. The exact install differs across Colab images; use the official torch-xla install snippet for “Colab TPU”.

Minimal pattern:

```bash
%%bash
set -euo pipefail

cd "$REPO_DIR"
python3 -m pip install -U pip

# 1) Install torch_xla (TPU) + matching torch build (per torch-xla docs)
# Example only: adjust versions if Colab images change.
# python3 -m pip install -U "torch==2.2.*" "torchvision==0.17.*" "torchaudio==2.2.*"
# python3 -m pip install -U "torch_xla[tpu]==2.2.*" -f https://storage.googleapis.com/libtpu-releases/index.html

# 2) Install project deps
python3 -m pip install -r code/requirements.txt

# Sanity: import check
python3 -c "import torch; import rl4co; print('torch', torch.__version__); print('rl4co', rl4co.__version__)"
```

If the above fails due to version conflicts, the rule is:
- **torch_xla and torch must match** (same major/minor).
- Reinstall torch/torch_xla first, then install `rl4co==0.6.0`.

#### Cell D — Set PYTHONPATH and output locations

```bash
%%bash
set -euo pipefail

export PYTHONPATH="$REPO_DIR/code"
mkdir -p "$DRIVE_ROOT"/{runs,eval,diag}
```

### 2.3 TPU smoke test (must pass before full grid)

Goal: verify that a single short training run executes on TPU end-to-end.

```bash
%%bash
set -euo pipefail

cd "$REPO_DIR"

# Tiny run (fast). If this fails on TPU, do not continue on TPU.
python3 -m src.experiments.train_tsp \
  --algorithm maxk_pomo \
  --num_loc 20 \
  --num_starts 16 \
  --k 4 \
  --variance_reduction subloo \
  --max_epochs 1 \
  --train_data_size 512 \
  --val_data_size 128 \
  --batch_size 64 \
  --accelerator tpu \
  --devices 8 \
  --precision 32-true \
  --output_dir "$DRIVE_ROOT/runs" \
  --run_name "smoke_tpu_maxk_tsp20_n16_k4_subloo_seed0" \
  --check_numerics
```

Expected outcome:
- A checkpoint exists under `$DRIVE_ROOT/runs/<run_name>/checkpoints/last.ckpt`
- A CSV log exists under `$DRIVE_ROOT/runs/<run_name>/.../metrics.csv`

If this fails due to XLA issues, repeat the exact same command on **Colab GPU** by changing:

- `--accelerator gpu --devices 1`
- `--precision 16-mixed` (optional, often faster on GPU)

---

## 3. Experiment design (what to compare)

### 3.1 Algorithms (primary comparisons)

We compare three training objectives:

1. `pomo` — standard shared-baseline REINFORCE (risk-neutral training)
2. `leader_reward` — heuristic “leader bonus” baseline
3. `maxk_pomo` — principled unbiased Max@K gradient estimator

### 3.2 Evaluation protocols (what we report)

For each trained checkpoint:

- **Greedy**: `--method greedy` (single decode)
- **Best-of-K sampling**: `--method sampling --k_eval K` (e.g., K ∈ {16, 64, 128})

We report avg cost (negative reward) and standard deviation:

- `code/src/experiments/evaluate.py` writes `avg_cost`, `avg_reward`, and `reward_std`.

### 3.3 Diagnostics (why the estimator is “better”)

For each checkpoint we run diagnostics:

- Gradient variance proxies across resampling replicates (std of grad norm / projection)
- Weight concentration via ESS-style metric

These are produced by:

- `code/src/experiments/diagnose_gradients.py`

---

## 4. Concrete run grid (Phase 4 tasks)

This is the order we run experiments to minimize wasted compute.

### 4.1 Stage A: pick stable defaults on TSP50 (T4.1 + T4.7 + T4.8)

**Purpose:** determine a good default `(n, k, variance_reduction)` region before scaling.

Recommended fixed settings:

- seeds: `{0, 1, 2}` (expand to 5 if variance is high)
- `n = num_starts`: start with 16, optionally 32 later
- training budget: start small (sanity), then scale (final)

Grid:

- `pomo`: no extra knobs
- `leader_reward`: `alpha ∈ {0.0, 0.5, 1.0}` (alpha=0 is a control)
- `maxk_pomo`:
  - `k ∈ {1, 2, 4, 8, 16}` (must satisfy `k <= n`)
  - `variance_reduction ∈ {none, sample_loo, subloo}`
    - Sample-LOO requires `n > k`
    - SubLOO requires `k >= 2`

Copy/paste command patterns (edit the run name so it’s unique):

```bash
# Baseline POMO
python3 -m src.experiments.train_tsp \
  --algorithm pomo \
  --seed 0 \
  --num_loc 50 \
  --num_starts 16 \
  --max_epochs 10 \
  --batch_size 128 \
  --train_data_size 20000 \
  --val_data_size 2000 \
  --accelerator tpu --devices 8 --precision 32-true \
  --output_dir "$DRIVE_ROOT/runs" \
  --run_name "tsp50_pomo_n16_seed0"

# Leader Reward
python3 -m src.experiments.train_tsp \
  --algorithm leader_reward \
  --seed 0 \
  --num_loc 50 \
  --num_starts 16 \
  --alpha 0.5 \
  --max_epochs 10 \
  --batch_size 128 \
  --train_data_size 20000 \
  --val_data_size 2000 \
  --accelerator tpu --devices 8 --precision 32-true \
  --output_dir "$DRIVE_ROOT/runs" \
  --run_name "tsp50_leader_alpha0.5_n16_seed0"

# Principled Max@K
python3 -m src.experiments.train_tsp \
  --algorithm maxk_pomo \
  --seed 0 \
  --num_loc 50 \
  --num_starts 16 \
  --k 8 \
  --variance_reduction subloo \
  --max_epochs 10 \
  --batch_size 128 \
  --train_data_size 20000 \
  --val_data_size 2000 \
  --accelerator tpu --devices 8 --precision 32-true \
  --output_dir "$DRIVE_ROOT/runs" \
  --run_name "tsp50_maxk_k8_subloo_n16_seed0" \
  --check_numerics
```

After Stage A, choose:

- 1–2 best `leader_reward` alphas
- 1–2 best `maxk_pomo` configurations (likely with SubLOO or Sample-LOO)

Criteria:

- Best evaluation `avg_cost` at a fixed `k_eval` (e.g., 128)
- Lower diagnostics variance / higher ESS (more stable training)
- Similar or lower wall-clock per step (overhead sanity)

### 4.2 Stage B: scale to TSP100 (T4.1 + T4.6)

Run only the best configs from Stage A on `num_loc=100`. Keep the comparison clean:

- `pomo`
- best `leader_reward`
- best `maxk_pomo`

Use the same seeds and evaluation protocol.

### 4.3 Stage C: other tasks (T4.2–T4.4)

PRD Phase 4 includes CVRP/OP/PCTSP. Today this repo only ships `train_tsp.py`.

Phase 4 engineering prerequisite:

1. Add training entrypoints for the missing tasks (either `train_cvrp.py` etc. or a generalized `train.py --problem ...`).
2. Run a tiny smoke test for each new task on TPU (same as §2.3).
3. Only then run the full experiment grid (Stage A → Stage B style) per task.

Deliverable expectation for each task:

- A table of best costs for greedy and best-of-K evaluation.
- Training curves (cost vs. epoch and cost vs. wall-clock).
- Diagnostics for at least 1 representative checkpoint per algorithm.

---

## 5. Evaluation + diagnostics runbook (per checkpoint)

Given a checkpoint:

```bash
CKPT="$DRIVE_ROOT/runs/<run_name>/checkpoints/last.ckpt"
```

### 5.1 Evaluation (T4.1–T4.4)

Greedy:

```bash
python3 -m src.experiments.evaluate \
  --problem tsp \
  --num_loc 50 \
  --algorithm maxk_pomo \
  --ckpt_path "$CKPT" \
  --method greedy \
  --num_instances 1000 \
  --device cpu \
  --batch_size 256 \
  --save_path "$DRIVE_ROOT/eval/tsp50_eval.jsonl"
```

Best-of-128 sampling:

```bash
python3 -m src.experiments.evaluate \
  --problem tsp \
  --num_loc 50 \
  --algorithm maxk_pomo \
  --ckpt_path "$CKPT" \
  --method sampling \
  --k_eval 128 \
  --num_instances 1000 \
  --device cpu \
  --batch_size 256 \
  --save_path "$DRIVE_ROOT/eval/tsp50_eval.jsonl"
```

Notes:
- Evaluating on TPU may require XLA-specific device handling; simplest is `--device cpu` or switch runtime to GPU for evaluation.
- Always evaluate baselines and Max@K models with the same `--method/--k_eval` for fairness.

### 5.2 Diagnostics (T4.5)

```bash
python3 -m src.experiments.diagnose_gradients \
  --problem tsp \
  --num_loc 50 \
  --algorithm maxk_pomo \
  --ckpt_path "$CKPT" \
  --num_instances 128 \
  --batch_size 64 \
  --num_replicates 8 \
  --device cpu \
  --save_path "$DRIVE_ROOT/diag/tsp50_diag.jsonl"
```

---

## 6. Bookkeeping (how we avoid chaos)

### 6.1 Naming convention (required)

The default `train_tsp.py` run name does not include `k` or variance reduction, so **always set `--run_name`**.

Recommended format:

```text
<problem><size>_<algorithm>_n<num_starts>_k<k-or-na>_<vr-or-alpha>_seed<seed>
```

Examples:

- `tsp50_pomo_n16_seed0`
- `tsp50_leader_alpha0.5_n16_seed1`
- `tsp50_maxk_k8_subloo_n16_seed2`

### 6.2 Output layout on Drive

```text
$DRIVE_ROOT/
  runs/   # Lightning logs + checkpoints
  eval/   # JSONL from evaluate.py
  diag/   # JSONL from diagnose_gradients.py
  analysis/  # (optional) notebooks / plotting scripts
```

---

## 7. Phase 4 “done” checklist

We consider Phase 4 complete when:

- TSP50 + TSP100 have complete tables + curves for `pomo`, `leader_reward`, `maxk_pomo`.
- At least one additional task beyond TSP is run end-to-end (training + evaluation + diagnostics).
- We have:
  - A `k` sweep plot/table (T4.7).
  - A variance-reduction ablation plot/table (T4.8).
  - A diagnostics figure showing reduced variance / improved ESS (T4.5).
  - A convergence comparison figure (T4.6).
