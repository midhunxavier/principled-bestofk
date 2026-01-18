# Task 2.3 — LOO Baselines for Max@K (Implementation Notes)

**Task:** T2.3 — Implement LOO variance reduction baseline  
**Status:** Complete  
**Date:** January 18, 2026  

---

## 1. Goal

Implement variance-reduction baselines that preserve unbiasedness for the Max@K policy gradient estimator:

```math
\widehat{G}_{n,K}(\theta) = \sum_{i=1}^{n} s_i\, \nabla_\theta \log \pi_\theta(\tau_i).
```

The baselines must satisfy the score-function condition: the subtracted baseline for sample \(i\) must be independent of \(\tau_i\).

Primary reference:

- `docs/Tasks/Task1/task1.3/loo_variance_reduction.md`

---

## 2. Two Baseline Methods

### 2.1 Sample-LOO (requires n > K)

Define the leave-one-out baseline:

```math
b_i^{\mathrm{LOO}} := \hat{\rho}^{(g)}(n-1, K; \text{excluding sample } i)
= \frac{1}{\binom{n-1}{K}} \sum_{\substack{S \subseteq [n]\setminus\{i\}\\|S|=K}} \max_{j\in S} R_j.
```

Then the unbiased variance-reduced estimator is:

```math
\boxed{\widehat{G}^{\mathrm{Sample\text{-}LOO}}_{n,K}(\theta)
= \sum_{i=1}^{n} (s_i - b_i^{\mathrm{LOO}}) \, \nabla_\theta\log \pi_\theta(\tau_i)}
```

Implementation note (critical):

- This is **not** `w_i * (R_i - b_i)`.
- Compute full gradient weights `s_i` (T2.2), then subtract the LOO baseline.

Closed forms for `b_{σ(i)}^{LOO}` are provided in `loo_variance_reduction.md` §2.1.1.

### 2.2 SubLOO (requires K ≥ 2)

Per-subset baseline for subset \(S\) and element \(i\in S\):

```math
b_{i,S} := \max_{j \in S\setminus\{i\}} R_j.
```

The unbiased, hitchhiking-free estimator:

```math
\boxed{\widehat{G}^{\mathrm{SubLOO}}_{n,K}(\theta)
= \frac{1}{\binom{n}{K}} \sum_{|S|=K} \sum_{i\in S}
(\max_{j\in S} R_j - b_{i,S})\,\nabla_\theta\log\pi_\theta(\tau_i)}
```

Key property:

- For a fixed subset \(S\), only the subset maximum receives a non-zero term, equal to the max–second-max gap.

Closed form (Proposition 2.1 in Task 1.3) gives the effective per-sample weights
\(\tilde{s}_i^{\mathrm{SubLOO}}\) in terms of reward gaps under sorting.

---

## 3. Code Location and API (PyTorch)

Implementation is in:

- `code/src/estimators/baselines.py`

Functions:

- `sample_loo_baseline(rewards: torch.Tensor, k: int, *, stable_sort: bool = True) -> torch.Tensor`
  - Returns `b_i^LOO` in original sample order.
- `apply_sample_loo(s: torch.Tensor, rewards: torch.Tensor, k: int, *, stable_sort: bool = True) -> torch.Tensor`
  - Returns `s - b_loo` in original sample order.
  - Must validate `n > k`.

- `subloo_weights(rewards: torch.Tensor, k: int, *, stable_sort: bool = True) -> torch.Tensor`
  - Returns hitchhiking-free weights \(\tilde{s}^{\mathrm{SubLOO}}\) (already baseline-adjusted), in original sample order.
  - Must validate `k >= 2`.

---

## 4. Test Strategy (T2.4 dependency)

Unit tests are in `code/tests/test_baselines.py`.

### Sample-LOO tests

- For small \((n,K)\), compute `b_i^{LOO}` by enumeration of all size-\(K\) subsets of `[n]\{i}`.
- Verify `(s_i - b_i^{LOO})` is correct relative to the definition.

### SubLOO tests

- Enumerate all subsets \(S\) and compute per-sample contributions:

```math
\tilde{s}_i^{\mathrm{SubLOO}} := \frac{1}{\binom{n}{K}}\sum_{\substack{|S|=K\\i\in S}}
(\max_{j\in S} R_j - \max_{j\in S\setminus\{i\}} R_j)
```

- Compare to the closed-form implementation.

---

## 5. Interaction with training code

- In the final loss, the weights (`s` or baseline-adjusted variants) must be used with stop-gradient / `.detach()` semantics:

```python
loss = -(weights.detach() * log_probs).sum(dim=-1).mean()
```

This avoids backpropagating through the sorting / combinatorics.
