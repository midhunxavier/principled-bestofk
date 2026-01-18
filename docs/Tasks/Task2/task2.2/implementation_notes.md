# Task 2.2 — Max@K Gradient Score Weights (Implementation Notes)

**Task:** T2.2 — Implement gradient weight computation  
**Status:** Planned / Notes updated  
**Date:** January 18, 2026  

---

## 1. Goal

Implement the **per-sample gradient score weights** \(s_i\) such that the unbiased Max@K policy gradient estimator can be written as a REINFORCE-style sum:

```math
\widehat{\nabla_\theta J}_{\mathrm{max}@K}
= \sum_{i=1}^{n} s_i \, \nabla_\theta \log \pi_\theta(\tau_i)
```

Critically, for \(K \ge 2\), these \(s_i\) **are not** the same as the reward-estimator weights \(w_i\) from T2.1.

---

## 2. Reference Derivation

Primary reference:

- `docs/Tasks/Task1/task1.2/unbiasedness_proof.md` (Proposition 5.1)

---

## 3. Definition (U-statistic form)

Given \(n \ge K\) i.i.d. trajectories \(\tau_{1:n}\), the U-statistic estimator is:

```math
\widehat{G}_{n,K}(\theta)
= \frac{1}{\binom{n}{K}}
\sum_{|S|=K}
\Big( \max_{j\in S} R(\tau_j) \Big)
\Big( \sum_{i\in S} \nabla_\theta\log\pi_\theta(\tau_i) \Big)
```

Swapping sums yields the per-sample form with:

```math
s_i
:=
\frac{1}{\binom{n}{K}} \sum_{\substack{|S|=K\\ i\in S}} \max_{j\in S} R(\tau_j).
```

---

## 4. Closed Form (order statistics)

Let \(\sigma\) sort rewards ascending:

```math
R_{\sigma(1)} \le \cdots \le R_{\sigma(n)},
\quad R_{(i)} := R_{\sigma(i)}.
```

For \(K \ge 2\), Proposition 5.1 gives:

```math
\boxed{
 s_{\sigma(i)}
 =
 \frac{1}{\binom{n}{K}}
 \left[
 \mathbf{1}[i\ge K]\binom{i-1}{K-1}R_{(i)}
 +
 \sum_{j=i+1}^{n}\binom{j-2}{K-2}R_{(j)}
 \right]
}
```

Interpretation:

- **Win term**: proper credit when sample \(i\) is the subset maximum.
- **Support term**: contributions from subsets where a higher-ranked sample wins (hitchhiking source).

Special cases:

- \(K=1\): standard REINFORCE weights are \(s_i = R_i / n\).
- \(K=n\): \(s_i = \max_j R_j\) for all \(i\).

---

## 5. Intended API (PyTorch)

Recommended module:

- `src/estimators/maxk_gradient.py` (or keep under `src/estimators/maxk_estimator.py` if consolidating)

Functions:

- `maxk_score_weights(rewards: torch.Tensor, k: int, *, stable_sort: bool = True) -> torch.Tensor`
  - Input rewards: `[n]` or `[batch, n]`
  - Output: weights `[n]` or `[batch, n]` aligned with original sample order

Implementation sketch:

1. Sort rewards ascending, keep permutation \(\sigma\)
2. Compute closed-form \(s_{\sigma(i)}\) in sorted order
3. Scatter back to original indices (inverse permutation)

Notes:

- Must use the full Proposition 5.1 formula; do **not** approximate using only win weights \(w_i\).

---

## 6. Test Strategy (T2.4 dependency)

- For small \((n,K)\), verify `maxk_score_weights` equals the exact enumeration definition:

```math
s_i = \frac{1}{\binom{n}{K}} \sum_{\substack{|S|=K\\i\in S}} \max_{j\in S} R_j
```

computed via explicit subset enumeration.

- Check special cases \(K=1\), \(K=n\).

---

## 7. Relation to T2.1 and T2.3

- Reward estimator: \(\hat{\rho}^{(g)}(n,K) = \sum_i w_i R_{(i)}\).
- Gradient estimator uses \(s_i\), not \(w_i\). Also:

```math
\sum_{i=1}^{n} s_i = K \cdot \hat{\rho}^{(g)}(n,K).
```

- Sample-LOO and SubLOO (T2.3) operate on **gradient weights** (subtract baselines from \(s_i\)), not on \(w_i\cdot(R_i-\cdot)\).
