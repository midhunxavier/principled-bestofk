# Task 2.4 — Unit Tests for Estimators (Implementation Notes)

**Task:** T2.4 — Unit tests for all components  
**Status:** Complete  
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

## 3. Implemented Coverage

- `code/tests/test_maxk_reward.py`: definition-by-enumeration checks for the Max@K reward estimator.
- `code/tests/test_maxk_gradient.py`: definition-by-enumeration checks for the Max@K gradient score weights.
- `code/tests/test_baselines.py`: enumeration checks for Sample-LOO baselines and SubLOO (winner-gap) weights.
- `code/tests/test_pkpo_rspo_validation.py`: independent closed-form references for PKPO/RSPO-style formulas.
- `code/tests/test_stability.py`: large-n finiteness / overflow-protection smoke tests.
- `code/tests/conftest.py`: ensures `code/` is on `sys.path` for imports.

---

## 4. Running tests

```bash
python3 -m pytest -v
```
