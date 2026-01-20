# Task 4.1 Failure Analysis and Revised Experimental Protocol

**Date:** January 20, 2026  
**Status:** Analysis Complete, Action Items Identified

---

## 1. Executive Summary

The initial T4.1 experiments showed that our principled Max@K (SubLOO) approach **significantly underperformed** both POMO and Leader Reward baselines:

| Algorithm | TSP50 Greedy Cost | TSP50 Best-of-128 Cost |
|-----------|-------------------|------------------------|
| POMO | 6.45 ± 0.05 | 5.98 ± 0.01 |
| Leader Reward | 6.53 ± 0.05 | 6.01 ± 0.01 |
| **MaxK POMO (SubLOO)** | **6.73 ± 0.17** | **6.28 ± 0.05** |

The training curves show `maxk_pomo` oscillating around 7.7-8.0 while baselines converge to ~6.4.

This document identifies **5 critical issues** in the implementation/design and proposes a **revised experimental protocol**.

---

## 2. Critical Issues Identified

### 2.1 ISSUE #1: Missing Weight Normalization (CRITICAL BUG)

**Location:** `code/src/algorithms/losses.py:203`

**Current code:**
```python
loss = -(weights.detach() * log_likelihood).sum(dim=-1).mean()
```

**Problem:** The SubLOO weights are **not normalized** and can have drastically different magnitudes than POMO/LeaderReward advantages:

- POMO advantage: `R_i - mean(R)` ~ O(0.1) for TSP with costs ~6.5
- SubLOO weights: Sum to `K * avg_gap` ~ O(1-2) but individual weights vary from 0 to O(1)

The gradient is being scaled by ~10-20x compared to baselines, causing:
- Overshooting in optimization
- Unstable training dynamics
- The oscillating validation curve we observed

**Compare with Leader Reward:** At `leader_reward.py:208`:
```python
advantage = self.advantage_scaler(advantage)  # Normalizes!
```

Leader Reward uses `self.advantage_scaler` (from RL4CO) but MaxKPOMO applies it only optionally via `scale_fn` which may not be configured.

**Fix:** Add proper normalization to SubLOO weights before computing loss:
```python
# Normalize to zero-mean, unit-std (like POMO does)
weights = (weights - weights.mean(dim=-1, keepdim=True)) / (weights.std(dim=-1, keepdim=True) + 1e-8)
```

### 2.2 ISSUE #2: SubLOO Weights Can Be Zero for All Samples (CRITICAL)

**Location:** `code/src/estimators/baselines.py:239-318`

**Problem:** When rewards are nearly identical (common in early training with a random policy), SubLOO gaps are ~0 for all samples:

```
gap = R_max - R_second_max ≈ 0 when policy is random
```

This means the gradient signal is **essentially zero**, preventing any learning.

**Evidence:** The maxk_pomo validation cost stays flat at ~7.7-8.0, indicating the model isn't learning.

**Fix Options:**
1. Add a minimum gap floor: `gap = max(gap, epsilon)`
2. Use a hybrid: SubLOO + small POMO baseline term
3. Add exploration bonus / entropy regularization

### 2.3 ISSUE #3: Gradient Clipping Was Disabled in Actual Run

**Location:** Manifest shows `gradient_clip_val: null`

From `artifacts_t4_1/manifest.json:109`:
```json
"gradient_clip_val": null
```

Despite the training script default being `1.0`, the actual run had it disabled. This may have allowed gradient explosions.

**Evidence:** The high variance across seeds (±0.17 for maxk vs ±0.05 for POMO) suggests unstable gradients.

**Fix:** Always enable gradient clipping: `--gradient_clip_val 1.0`

### 2.4 ISSUE #4: n/k Ratio Too Aggressive

**Configuration:** `n=16, k=8` (ratio = 2.0)

**Problem:** This is at the minimum recommended ratio. From `docs/Tasks/Task1/task1.3:349`:
> Use $n \geq 2K$ if possible

With k=8 and n=16:
- SubLOO only gives gradient to the top-8 samples
- Bottom 8 get exactly 0 gradient
- This is too sparse for stable learning

**Fix:** Use `n=32, k=8` (ratio 4.0) or `n=16, k=4` (ratio 4.0)

### 2.5 ISSUE #5: Only 10 Epochs (Too Short)

**Configuration:** `max_epochs=10`

Standard POMO training uses **100+ epochs**. With sparse SubLOO gradients, even more epochs may be needed.

**Fix:** Train for at least 50-100 epochs.

---

## 3. Additional Design Concerns

### 3.1 POMO's Deterministic Multi-Start Violates i.i.d. Assumption

The theoretical proofs in Task 1 assume **i.i.d. samples**. POMO's deterministic multi-start (one sample per starting node) produces **exchangeable but not i.i.d.** samples.

**Impact:** The unbiasedness guarantee may not hold strictly.

**Mitigation:** This is a known limitation acknowledged in PRD:146-148. For strict i.i.d., sample start nodes randomly with replacement.

### 3.2 No Entropy Regularization

SubLOO eliminates gradient signal to non-top-k samples. Without entropy regularization, the policy can collapse to deterministic behavior quickly, losing exploration.

**Mitigation:** Add `entropy_coef > 0` (e.g., 0.01-0.1).

### 3.3 reward_scale Not Configured

The training didn't use any reward scaling (`reward_scale: null`), but for TSP the costs are O(6-8). This is fine for POMO (which uses relative advantages) but may interact poorly with SubLOO's gap-based weights.

---

## 4. Revised Experimental Protocol

### 4.1 Bug Fixes Required Before Re-running

| Priority | Issue | Fix Location | Action |
|----------|-------|--------------|--------|
| P0 | Weight normalization | `losses.py:MaxKLoss.__call__` | Add z-score normalization |
| P0 | Zero-gradient early training | `baselines.py:subloo_weights` | Add minimum gap floor or hybrid |
| P1 | Gradient clipping | CLI/config | Enforce `--gradient_clip_val 1.0` |

### 4.2 Recommended Configuration Changes

```bash
# Revised Phase 4 run command
python3 -m src.experiments.phase4_tsp_experiments train \
  --runs_dir "$RUNS_DIR" \
  --artifacts_dir "$ARTIFACTS_DIR" \
  --num_locs 50 \
  --seeds 0 1 2 \
  --algorithms pomo leader_reward maxk_pomo \
  --num_starts 32 \          # Increased from 16 -> better n/k ratio
  --maxk_k 8 \
  --maxk_variance_reduction sample_loo \  # Try sample_loo first (less aggressive)
  --max_epochs 100 \         # Increased from 10
  --batch_size 64 \          # Reduced for memory with n=32
  --train_data_size 100000 \ # Increased training data
  --gradient_clip_val 1.0 \  # Explicitly enabled
  --lr 1e-4 \
  --accelerator gpu \
  --devices 1
```

### 4.3 Ablation Experiments to Run

| Experiment | Purpose | Configuration |
|------------|---------|---------------|
| A1: Sample-LOO vs SubLOO | Compare variance reduction methods | `--variance_reduction sample_loo` vs `subloo` |
| A2: n/k ratio sweep | Find optimal ratio | n=16,k=4; n=32,k=8; n=64,k=8 |
| A3: Weight normalization | Verify fix works | With/without z-score normalization |
| A4: Hybrid baseline | Combine SubLOO + POMO | `0.5 * SubLOO + 0.5 * POMO_advantage` |
| A5: Entropy regularization | Exploration | `--entropy_coef 0.01` |

### 4.4 Success Criteria

Before declaring success, verify:

1. **Training stability:** MaxK validation cost should decrease monotonically (like POMO)
2. **Convergence:** Should reach within 10% of POMO performance
3. **Variance:** Seed-to-seed variance should be comparable to POMO

---

## 5. Code Changes Required

### 5.1 Fix Weight Normalization in `losses.py`

```python
# In MaxKLoss.__call__, after computing weights:

def __call__(
    self,
    rewards: torch.Tensor,
    log_likelihood: torch.Tensor,
    *,
    scale_fn: ScaleFn | None = None,
    normalize_weights: bool = True,  # NEW PARAMETER
) -> MaxKLossOutput:
    ...
    weights, rho_hat = self.compute_weights(rewards)
    
    # NEW: Normalize weights to stabilize training
    if normalize_weights:
        w_mean = weights.mean(dim=-1, keepdim=True)
        w_std = weights.std(dim=-1, keepdim=True)
        weights = (weights - w_mean) / (w_std + 1e-8)
    
    if scale_fn is not None:
        weights = scale_fn(weights)
    ...
```

### 5.2 Add Minimum Gap Floor in `baselines.py`

```python
# In subloo_weights, after computing sum_gaps:

def subloo_weights(
    rewards: torch.Tensor,
    k: int,
    *,
    stable_sort: bool = True,
    min_gap_scale: float = 0.01,  # NEW PARAMETER
) -> torch.Tensor:
    ...
    # After: sum_gaps = sorted_rewards * num_subsets - weighted_sum
    
    # NEW: Ensure minimum gradient signal
    reward_range = sorted_rewards[..., -1] - sorted_rewards[..., 0]
    min_gap = min_gap_scale * reward_range.unsqueeze(-1)
    sum_gaps = torch.maximum(sum_gaps, min_gap * (ranks >= k).float())
    ...
```

### 5.3 Add Hybrid Mode Option

Create a new variance reduction mode: `hybrid_subloo` that combines:
```python
weights = lambda_ * subloo_weights + (1 - lambda_) * pomo_advantage
```

This provides the best of both worlds: unbiased Max@K signal + stable POMO gradient.

---

## 6. Root Cause Summary

| Symptom | Root Cause | Category |
|---------|------------|----------|
| No convergence | Zero/tiny SubLOO weights in early training | Design flaw |
| High variance | Unnormalized weights + no gradient clipping | Implementation bug |
| Worse than POMO | Weights not comparable in scale to advantages | Implementation bug |
| Oscillating training | Gradient explosions/vanishing | Configuration error |

**Bottom line:** The math is correct, but the engineering failed to account for practical considerations that make the theoretical estimator work in a real training loop.

---

## 7. Next Steps

1. [ ] Implement weight normalization fix
2. [ ] Implement minimum gap floor
3. [ ] Re-run TSP50 with `sample_loo` and n=32, k=8, 100 epochs
4. [ ] Compare training curves
5. [ ] If still failing, implement hybrid mode
6. [ ] Run full ablation suite

---

**Author:** Analysis by AI Agent  
**Review Status:** Pending human review
