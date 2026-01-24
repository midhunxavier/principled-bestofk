# Task 4.1 — Analysis (Part 2): Hybrid Max@K vs Baselines (TSP50)

**Date:** 2026-01-23  
**Scope:** Follow-up analysis on Task 4.1 after stabilizing `maxk_pomo` with Hybrid variance reduction.

This report focuses on the concrete execution artifacts in:
- Runs: `docs/Tasks/Task4/Analysis-2/runs_t4_1/`
- Tables/plots: `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/`

---

## 1) What was executed (and what actually finished)

### 1.1 Training grid (TSP50)

All runs share:

- **Problem:** TSP50
- **Seeds:** `{0, 1}`
- **Rollouts:** `num_starts = 64`
- **Data:** train=20,000, val=2,000, test=2,000
- **Batch size:** 128
- **Optimizer:** Adam, `lr = 1e-4`
- **Epochs:** intended 50 (0–49)

Algorithms:

- `pomo` (RL4CO baseline)
- `leader_reward` with `alpha = 0.5`
- `maxk_pomo` with:
  - `k = 8`
  - `variance_reduction = hybrid`
  - `hybrid_lambda = 0.5`
  - `weight_normalization = zscore`
  - `min_gap_scale = 0.01`

### 1.2 Critical execution deviation: one Max@K run is truncated

The run `tsp50_maxk_k8_hybrid_n64_seed1` stopped around **epoch 20**:

- Checkpoints present: `019.ckpt` and `last.ckpt`
- `metrics.csv` contains validation up to **epoch 19** (no epoch 49 entry).

This matters because the evaluation table averages seeds `{0,1}` but uses a **partially-trained** Max@K model for seed 1, while the baselines have fully-trained checkpoints through epoch 49.

---

## 2) Results (evaluation on 1,000 fresh test instances)

Evaluation configuration:
- **Dataset:** generated test set (RL4CO env; seed `1234`)
- **Device:** CPU
- **Metrics:**
  - `greedy`: single-start greedy decoding
  - `sampling (k=128)`: best-of-128 sampling (randomized start nodes)

### 2.1 Aggregate table (mean ± std over seeds)

From `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/tables/t4_1_tsp_results.md`:

| algorithm | greedy cost (↓) | best-of-128 cost (↓) |
|---|---:|---:|
| `pomo` | 5.9854 ± 0.0232 | 5.7727 ± 0.0143 |
| `leader_reward` | 6.1449 ± 0.0742 | 5.7871 ± 0.0066 |
| `maxk_pomo` | 6.2100 ± 0.0014 | 5.8277 ± 0.0346 |

Per-seed rows (for transparency) live at:
`docs/Tasks/Task4/Analysis-2/artifacts_t4_1/tables/t4_1_tsp_eval_rows.csv`.

### 2.2 Key observations

1) **POMO remains the strongest** on both greedy and best-of-128 sampling in this run.

2) **Leader Reward trades off greedy quality for tail quality**:
- Greedy is meaningfully worse than POMO (+0.16 cost), but best-of-128 is very close (+0.014).

3) **Max@K Hybrid behaves like a “high-variance tail-chaser”, but doesn’t win yet**:
- Greedy is the worst of the three.
- Best-of-128 is closer to the baselines than greedy suggests, but still behind POMO.

4) **Sampling gains (diversity proxy)** differ across methods:

Let Δ = (greedy_cost − best_of_128_cost); larger Δ means “sampling rescues you more”.

- `pomo`: Δ ≈ 0.213
- `leader_reward`: Δ ≈ 0.358
- `maxk_pomo`: Δ ≈ 0.382

This pattern is consistent with Leader/Max@K shifting probability mass toward a heavier right tail (better rare samples), while sacrificing typical/greedy quality.

5) **Do not over-interpret Max@K seed variance** yet: the worse Max@K seed is also the truncated run.

---

## 3) Training-curve reading (validation cost)

Plot: `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/plots/t4_1_tsp50_val_cost.png`.

Interpretation (qualitative, because of the truncated seed):

- All methods show the expected rapid early improvement, then slower refinement.
- `pomo` is consistently the lowest curve (best validation cost).
- `leader_reward` tracks above `pomo`.
- `maxk_pomo` is comparable to `leader_reward` early, but is not clearly better at convergence in this run.

Important nuance:

- RL4CO POMO validation uses **multistart greedy** internally (best-of-`num_starts`), while our evaluation “greedy” metric is **single-start** greedy.
- This makes the training-curve values *systematically optimistic* relative to the “greedy” evaluation table, but it affects all algorithms similarly.

---

## 4) Gradient/weight diagnostics (why Max@K can be hard to optimize)

I ran `code/src/experiments/diagnose_gradients.py` on seed 0 checkpoints (64 instances, 4 replicates, CPU) to compare raw gradient/weight behavior:

| algorithm | grad_norm mean±std | weights_abs_max mean±std |
|---|---:|---:|
| `pomo` | 2.20 ± 0.17 | 0.68 ± 0.04 |
| `leader_reward` | 2.49 ± 0.25 | 0.87 ± 0.05 |
| `maxk_pomo` (hybrid) | 563.82 ± 57.16 | 2.87 ± 0.05 |

Files:
- `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/diag/tsp50_pomo_seed0.json`
- `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/diag/tsp50_leader_seed0.json`
- `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/diag/tsp50_maxk_seed0.json`

How to read this:

- The **Max@K loss has a very different scale** from the POMO/Leader losses (partly because it uses a `sum(dim=-1)` over `n` rollouts, and because weight normalization changes the natural scale of the unbiased estimator).
- If training relies on **gradient clipping**, these raw norms may not translate directly to update magnitudes; if clipping is *not* active, this strongly suggests Max@K needs a smaller LR or explicit loss scaling.

This provides a plausible mechanism for “stable but not better”: updates may be dominated by clipping / scale artifacts rather than by an informative objective signal.

---

## 5) Critical review (what this run does and does not prove)

### 5.1 Experimental integrity

- **Truncated Max@K seed** breaks the intended apples-to-apples comparison.
- **Only 2 seeds** were run; this is below the spec (Task 4.1 suggests ≥3 seeds).
- `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/manifest.json` does **not** record the full training args (many are zeroed), so the run provenance is not fully captured in a single place.

### 5.2 “Principled” claim vs stabilization knobs

The configuration used here is *not* “pure unbiased Max@K” in the strict sense:

- `variance_reduction = hybrid` explicitly mixes in a POMO-style advantage term.
- `weight_normalization = zscore` and `min_gap_scale > 0` change the estimator further.

These knobs are probably necessary for learning stability, but they should be reported as such (they change the target objective / estimator properties).

### 5.3 Objective / protocol mismatch (training vs evaluation)

- Training uses RL4CO’s **multistart** protocol, where start nodes are chosen deterministically via `env.select_start_nodes` (wrapping modulo `num_loc` when `num_starts > num_loc`).
- Best-of-128 evaluation uses **multisample sampling** with *random* start nodes.

This mismatch is not necessarily “wrong”, but it makes it harder to attribute wins/losses to the Max@K estimator itself rather than to start-node policy interactions.

---

## 6) Recommendations (high-leverage next steps)

### 6.1 Fix comparability and rerun the minimal grid

1) Rerun `tsp50_maxk_k8_hybrid_n64_seed1` to full 50 epochs (or rerun all seeds to 100 epochs, see below).
2) Add seed 2 for all three algorithms.
3) Ensure the “best-of-128” table is computed from **completed** runs only.

### 6.2 Scale to the Phase 4 plan (to answer the real question)

To test whether Max@K helps at meaningful compute:

- Increase training to **100 epochs** and **100k** training instances (matches `docs/Tasks/Task4/phase4_V1_execution_plan.md`).
- Consider setting `num_starts` to `50` (match TSP50) unless there is a specific reason to exceed it (wrap-around duplicates reduce effective start diversity).

### 6.3 Ablations that directly test the Max@K hypothesis

Keep everything else fixed (same seeds, same N, same data, same epochs):

- Sweep `hybrid_lambda ∈ {0.5, 0.8, 1.0}`
- Sweep `k ∈ {8, 16, 32}` for `n=64` (or `k ∈ {8, 12, 16}` for `n=50`)
- Compare `hybrid` vs `sample_loo` vs `subloo` with the *same* normalization strategy

### 6.4 Make “tail improvement” measurable

In addition to mean cost:

- Report quantiles (e.g., p50/p90/p99 of per-instance best-of-K costs).
- Track policy entropy / action entropy during training (Max@K-like objectives can collapse exploration).

---

## Appendix: Artifact index

- Evaluation results: `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/tables/t4_1_tsp_results.md`
- Per-run eval rows: `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/tables/t4_1_tsp_eval_rows.csv`
- Training curves CSV: `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/tables/t4_1_tsp_training_curves.csv`
- Training curve plot: `docs/Tasks/Task4/Analysis-2/artifacts_t4_1/plots/t4_1_tsp50_val_cost.png`

