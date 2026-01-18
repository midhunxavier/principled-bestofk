# Task 2.4 — Unit Tests for Estimators (Implementation Notes)

**Task:** T2.4 — Unit tests for all components  
**Status:** In progress / Notes updated  
**Date:** January 18, 2026  

---

## 1. Goal

Provide a small, self-contained test suite validating:

- T2.1: Unbiased Max@K reward estimator \(\hat{\rho}^{(g)}(n,K)\)
- T2.2: Unbiased gradient score weights \(s_i\)
- T2.3: Variance-reduced baselines:
  - Sample-LOO: \(s_i - b_i^{\mathrm{LOO}}\)
  - SubLOO: hitchhiking-free gap weights

Tests should be independent of RL4CO training loops and rely only on PyTorch + combinatorial enumeration.

---

## 2. Philosophy

For each estimator, verify equality against a **definition-by-enumeration** reference implementation for small \(n\) (e.g. \(n\le 8\)). This gives extremely strong correctness guarantees and catches off-by-one errors in combinatorics.

---

## 3. Existing coverage (T2.1)

Implemented in:

- `tests/test_maxk_estimator.py`

It verifies `maxk_reward_estimate` against:

```math
\frac{1}{\binom{n}{K}}\sum_{S:|S|=K}\max_{i\in S} R_i.
```

Also covers edge cases \(K=1\) and \(K=n\).

---

## 4. Planned tests (T2.2)

Create e.g. `tests/test_maxk_gradient_weights.py`.

Reference enumeration:

```math
s_i
= \frac{1}{\binom{n}{K}} \sum_{\substack{|S|=K\\ i\in S}} \max_{j\in S} R_j.
```

Checks:

- Closed-form weights (Prop. 5.1) match enumeration
- Special cases \(K=1\) and \(K=n\)
- Batch support

---

## 5. Planned tests (T2.3)

### Sample-LOO

Create e.g. `tests/test_maxk_sample_loo.py`.

Reference:

```math
b_i^{\mathrm{LOO}} = \frac{1}{\binom{n-1}{K}}\sum_{\substack{S\subseteq[n]\setminus\{i\}\\|S|=K}}\max_{j\in S} R_j.
```

Then assert implementation matches `(s_i - b_i^{LOO})`.

### SubLOO

Create e.g. `tests/test_maxk_subloo.py`.

Reference:

```math
\tilde{s}_i^{\mathrm{SubLOO}} =
\frac{1}{\binom{n}{K}}\sum_{\substack{|S|=K\\i\in S}}
\Big(\max_{j\in S} R_j - \max_{j\in S\setminus\{i\}} R_j\Big).
```

Also verify the “no hitchhiking” property empirically:

- In each subset \(S\), only argmax has nonzero contribution.

---

## 6. Suggested test parameters

- Use fixed seeds for determinism
- Use `dtype=torch.float64` for tight tolerances (`atol ~ 1e-10`)
- Small configs to enumerate:
  - \((n,K) \in \{(4,2),(5,2),(6,3),(8,4)\}\)

---

## 7. Running tests

```bash
python3 -m pytest -v
```
