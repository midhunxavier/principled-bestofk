# Task 2.5 — Validate against PKPO/RSPO formulas (Numerical Verification)

**Task:** T2.5 — Validate against PKPO/RSPO formulas  
**Status:** Complete  
**Date:** January 18, 2026  

---

## 1. Goal

Provide an explicit numerical validation that the implemented estimators in `code/src/estimators/` match the closed-form expressions used in PKPO/RSPO-style Max@K / best-of-K objectives.

This is distinct from T2.4 (definition-by-enumeration unit tests): T2.5 checks *formula equivalence* by implementing independent reference versions of the closed forms and comparing outputs.

---

## 2. Formulas validated

### 2.1 PKPO-style Max@K reward estimator (continuous maxg@k)

Let rewards be sorted ascending:

```math
R_{(1)} \le \cdots \le R_{(n)}.
```

The unbiased Max@K reward estimator (Task 1.1 / Theorem 3.1) is:

```math
\boxed{\hat{\rho}^{(g)}(n, K)
= \frac{1}{\binom{n}{K}} \sum_{i=K}^{n} \binom{i-1}{K-1} R_{(i)}.}
```

---

### 2.2 RSPO-style Max@K gradient score weights (order-statistic closed form)

The unbiased Max@K policy gradient can be written:

```math
\widehat{\nabla_\theta J}_{\mathrm{max}@K}
= \sum_{i=1}^{n} s_i \nabla_\theta \log \pi_\theta(\tau_i).
```

For `K ≥ 2`, the per-sample score weights are (Task 1.2 / Proposition 5.1):

```math
\boxed{
s_{\sigma(i)}
=
\frac{1}{\binom{n}{K}}
\left[
\mathbf{1}[i\ge K]\binom{i-1}{K-1}R_{(i)}
\;+\;
\sum_{j=i+1}^{n}\binom{j-2}{K-2}R_{(j)}
\right].
}
```

Also validated:

```math
\sum_{i=1}^n s_i = K \cdot \hat{\rho}^{(g)}(n,K).
```

---

### 2.3 PKPO-style Sample-LOO baseline (closed form)

For `n > K`, Sample-LOO defines:

```math
b_i^{\mathrm{LOO}}
=
\frac{1}{\binom{n-1}{K}}
\sum_{\substack{S \subseteq [n]\setminus\{i\}\\|S|=K}}
\max_{j\in S} R_j.
```

The closed-form expression in terms of sorted rewards is implemented and validated against `sample_loo_baseline` (Task 1.3 §2.1.1).

---

### 2.4 RSPO-style SubLOO / marginal contribution weights (closed form)

For `K ≥ 2`, SubLOO yields hitchhiking-free “marginal contribution” weights
(winner-only, max–second-max gap per subset). The closed form (Task 1.3 / Proposition 2.1) is:

```math
\boxed{
\tilde{s}_{\sigma(i)}^{\mathrm{SubLOO}}
=
\frac{1}{\binom{n}{K}}
\sum_{m=K}^{i}
\binom{m-2}{K-2}\,(R_{(i)}-R_{(m-1)})\cdot \mathbf{1}[i\ge K].
}
```

---

## 3. Code and tests

Core implementations:

- `code/src/estimators/maxk_estimator.py`
- `code/src/estimators/maxk_gradient.py`
- `code/src/estimators/baselines.py`

Formula-validation tests (independent closed-form reference implementations):

- `code/tests/test_pkpo_rspo_validation.py`

Run:

```bash
python3 -m pytest code/tests/test_pkpo_rspo_validation.py -v
```

---

## 4. Notes

- Tests use `torch.float64` and tight tolerances (`atol=1e-12`) to catch off-by-one combinatorial errors.
- Sorting uses `torch.sort(..., stable=True)` (deterministic tie-breaking) consistent with the estimator implementations.

