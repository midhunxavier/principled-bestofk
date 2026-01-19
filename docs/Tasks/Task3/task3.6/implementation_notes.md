# Task 3.6 — Implementation Notes (Gradient diagnostics harness)

**Task:** T3.6 — Gradient diagnostics harness (variance/ESS diagnostics scripts)  
**Status:** Implemented (script + smoke tests)  
**Last updated:** 2026-01-19  

---

## 1. Deliverables

- Diagnostics script: `code/src/experiments/diagnose_gradients.py`
  - Computes gradient variance proxies across repeated resampling (same instances, different RNG seeds)
  - Computes ESS-style concentration metrics for per-sample weights/advantages
  - Saves structured results (`.jsonl`, `.json`, or `.pkl`)
- Helper functions: `code/src/algorithms/losses.py` (`effective_sample_size`, `MaxKLoss`)
- Unit tests: `code/tests/test_diagnose_gradients_script.py`
  - End-to-end smoke: create a tiny `MaxKPOMO` checkpoint, run diagnostics with a few replicates, validate output schema

---

## 2. What is measured (and why)

This harness is intended to support the project goal of **principled, variance-reduced estimators**, not just end performance.

It reports:

- **Gradient variance proxy** across `num_replicates`:
  - gradient norm statistics
  - a fixed random-projection statistic (stable across runs) to capture directional variability without storing full gradients
- **Weight/advantage concentration diagnostics**:
  - mean ESS-style metric per replicate (averaged over instances)
  - mean max absolute weight/advantage per replicate

These metrics are meant to complement inference evaluation (`T3.5`), and are useful for Phase 4 ablations (e.g., `none` vs `sample_loo` vs `subloo`).

---

## 3. Implementation overview

### 3.1 Replicate protocol

For each replicate `r = 0..num_replicates-1`:

1. Set RNG seed to `seed + r` (Lightning + torch + numpy).
2. Run a forward pass with `phase="train"` to obtain:
   - `reward` `[batch, n]`
   - `log_likelihood` `[batch, n]`
3. Construct the algorithm-specific loss and weights/advantages:
   - `maxk_pomo`: uses `model.maxk_loss(...)` → `weights`
   - `pomo`: uses shared-baseline advantage `reward - mean(reward)` → `advantage`
   - `leader_reward`: same as POMO but adds the leader bonus term
4. Backpropagate a single scalar loss (mean over batches).
5. Record:
   - `grad_norm`: ℓ2 norm over policy parameters
   - `grad_proj`: dot product with a fixed random projection vector (fixed once per run)
   - `ess_mean`: mean ESS metric over instances
   - `weights_abs_max_mean`: mean max |weight| over instances

### 3.2 ESS diagnostic

The ESS-style metric is:

```math
\\mathrm{ESS}(w) = \\frac{(\\sum_i w_i)^2}{\\sum_i w_i^2}
```

Notes:

- This is used as a *diagnostic* for concentration/cancellation; policy-gradient weights can be signed, so this is not a strict importance-sampling ESS.
- Values near 1 indicate “single-sample dominated” weights; values closer to `n` indicate more evenly spread weights (up to cancellation effects).

---

## 4. Output schema

Results are written as a single record with `schema_version=1` containing:

- `config`: the full CLI config (with paths serialized as strings)
- `metrics`: summary statistics across replicates
- `per_replicate`: raw per-replicate series for `grad_norm`, `grad_proj`, `ess_mean`, `weights_abs_max_mean`

---

## 5. Usage

Example:

- Diagnostics for gradient stability + ESS (small run):
  - `python3 code/src/experiments/diagnose_gradients.py --problem tsp --num_loc 20 --algorithm maxk_pomo --ckpt_path /path/to/last.ckpt --num_instances 128 --batch_size 64 --num_replicates 8 --seed 1234 --device cpu --save_path .tmp/diag/tsp20_maxk_grad.json`

