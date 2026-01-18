# Task 2.1 — Max@K Reward Estimator (Implementation Notes)

**Task:** T2.1 — Implement MaxK reward estimator  
**Status:** Complete  
**Date:** January 17, 2026  

---

## 1. Goal

Implement an **unbiased estimator** of the Max@K objective value

```math
J_{\mathrm{max}@K}(\theta) = \mathbb{E}_{\tau_{1:K} \sim \pi_\theta}\,[\max_{j \in [K]} R(\tau_j)]
```

using **`n \ge K` samples** from the policy.

This task delivers the *reward estimator* \(\hat{\rho}^{(g)}(n, K)\). Gradient weights and variance reduction are handled in later tasks (T2.2/T2.3).

---

## 2. Mathematical Definition

Let rewards be sorted in ascending order (order statistics):

```math
R_{(1)} \le R_{(2)} \le \cdots \le R_{(n)}.
```

The unbiased Max@K reward estimator from Task 1.1 (Theorem 3.1) is:

```math
\boxed{\hat{\rho}^{(g)}(n, K)
= \frac{1}{\binom{n}{K}} \sum_{i=K}^{n} \binom{i-1}{K-1} R_{(i)} }
```

Equivalently, define weights (1-indexed rank \(i\)):

```math
w_i = \mathbf{1}[i \ge K] \cdot \frac{\binom{i-1}{K-1}}{\binom{n}{K}},
\qquad
\hat{\rho}^{(g)}(n,K) = \sum_{i=1}^{n} w_i R_{(i)}.
```

**Sanity checks:**

- \(K=1\) (risk-neutral): \(\hat{\rho}^{(g)}(n,1)=\frac{1}{n}\sum_i R_i\)
- \(K=n\): \(\hat{\rho}^{(g)}(n,n)=\max_i R_i\)

---

## 3. Code Location and API

Implementation is in:

- `code/src/estimators/maxk_reward.py`

Public functions:

- `maxk_reward_weights(n: int, k: int, *, device=None, dtype=None) -> torch.Tensor`
  - Returns a tensor of shape `[n]` of order-statistic weights (0-indexed ranks).
- `maxk_reward_estimate(rewards: torch.Tensor, k: int, *, stable_sort: bool = True) -> torch.Tensor`
  - Supports rewards shaped `[n]` (returns scalar) or `[batch, n]` (returns `[batch]`).

Tie-handling:

- Sorting uses `torch.sort(..., stable=True)` by default, which provides deterministic tie-breaking by original index.

---

## 4. Tests

Unit tests are in:

- `code/tests/test_maxk_estimator.py`

The key correctness check is an **exact enumeration** reference:

```math
\frac{1}{\binom{n}{K}} \sum_{S:|S|=K} \max_{i\in S} R_i
```

computed via `itertools.combinations` for small `n`, and compared against `maxk_reward_estimate`.

To run:

```bash
python3 -m pytest code/tests/test_maxk_estimator.py -v
```

---

## 5. Notes / Follow-ups

- This task only implements the **reward estimator weights** \(w_i\).  
  The **gradient score-weights** \(s_i\) are different for \(K \ge 2\) and must include the Support term (see Task 1.2 / Proposition 5.1). That is T2.2.
- Sample-LOO and SubLOO baselines (T2.3) will depend on the reward estimator and sorting logic established here.
