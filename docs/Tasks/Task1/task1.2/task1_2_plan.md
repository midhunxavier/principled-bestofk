# Task T1.2 Plan — Prove Unbiasedness Property (Theorem + Proof)

**Implementation status:** Complete — proof drafted in `docs/Tasks/Task1/task1.2/unbiasedness_proof.md` and linked from `docs/Tasks/Task1/task1.1/mathematical_derivation.md`.

## 1. Objective

Produce a rigorous **theorem + proof** that the proposed Max@K / Best-of-K policy-gradient estimator is **unbiased**, i.e.

$$
\mathbb{E}[\widehat{\nabla_\theta J}_{\mathrm{max}@K}(\theta)] \;=\; \nabla_\theta J_{\mathrm{max}@K}(\theta),
\qquad
J_{\mathrm{max}@K}(\theta) \;=\; \mathbb{E}_{\tau_{1:K}\stackrel{\mathrm{i.i.d.}}{\sim}\pi_\theta}\big[\max_{j\in[K]} R(\tau_j)\big].
$$

This task should close the current “proof sketch” gap in `docs/Tasks/Task1/task1.1/mathematical_derivation.md` (Section 5.2).

---

## 2. Inputs / References

- `docs/PRD.md` (Task table; objective definition; intended estimator shape)
- `docs/Tasks/Task1/task1.1/mathematical_derivation.md` (Theorem 4.1 statement + sketch; notation for sorting permutation `σ`)
- (Optional) `knowledgebase/` references (PKPO/RSPO) for proof style and “marginal contribution” framing

---

## 3. Deliverables (Planned)

- Primary deliverable: `docs/Tasks/Task1/task1.2/unbiasedness_proof.md` (done)
  - Clear setup + assumptions
  - Theorem statement(s)
  - Full proof (lemma-based)
  - Appendix: combinatorial identities, tie-handling, special cases
- Secondary deliverable: update `docs/Tasks/Task1/task1.1/mathematical_derivation.md` to replace/shorten the proof sketch and link to the detailed proof (done).

---

## 4. Theorem to Prove (Draft Target Statement)

Let $\tau_1,\dots,\tau_n \stackrel{\mathrm{i.i.d.}}{\sim}\pi_\theta$ with $n\ge K$ and define the symmetric kernel

$$
h(\tau_{i_1},\dots,\tau_{i_K}) \;=\; \max_{m\in[K]} R(\tau_{i_m}).
$$

Define the **U-statistic gradient estimator**:

$$
\widehat{G}_{n,K}(\theta)
\;=\;
\frac{1}{\binom{n}{K}}
\sum_{\substack{S\subseteq[n]\\|S|=K}}
h(\tau_S)\;\sum_{j\in S}\nabla_\theta\log \pi_\theta(\tau_j).
$$

**Theorem (Unbiasedness).**
Under standard regularity assumptions (exchange of $\nabla_\theta$ and expectation; integrability of $R$ and score function),

$$
\mathbb{E}\big[\widehat{G}_{n,K}(\theta)\big] \;=\; \nabla_\theta J_{\mathrm{max}@K}(\theta).
$$

**Implementation form (equivalence).**
Let $\sigma$ sort rewards ascending, $R_{\sigma(1)}\le\cdots\le R_{\sigma(n)}$. Then $\widehat{G}_{n,K}$ can be rewritten as

$$
\widehat{G}_{n,K}(\theta)=\sum_{i=1}^n s_i\;\nabla_\theta\log \pi_\theta(\tau_{\sigma(i)}),
$$

for coefficients $s_i$ computable from $\{R_{\sigma(1)},\dots,R_{\sigma(n)}\}$ via combinatorial counting (spelled out in the proof).

---

## 5. Proof Roadmap (What the Proof Document Should Contain)

### 5.1 Setup + Assumptions (Make Them Explicit)

- $\pi_\theta(\tau)$ is differentiable in $\theta$ and either a density or mass function over trajectories.
- Score function exists: $\nabla_\theta \log \pi_\theta(\tau)$.
- Exchange of gradient and expectation is justified (e.g., dominated convergence / integrability conditions).
- Rewards have adequate integrability (bounded or $\mathbb{E}[|R(\tau)|]<\infty$).
- Tie-handling: assume $R(\tau)$ is almost surely distinct, or define a deterministic tie-break rule / $\epsilon$-perturbation.

### 5.2 Lemma A — Score-Function Form of the Max@K Gradient

Show the identity:

$$
\nabla_\theta J_{\mathrm{max}@K}(\theta)
\;=\;
\mathbb{E}_{\tau_{1:K}\stackrel{\mathrm{i.i.d.}}{\sim}\pi_\theta}\!\left[
\max_{j\in[K]} R(\tau_j)\;
\sum_{m=1}^K \nabla_\theta \log \pi_\theta(\tau_m)
\right].
$$

This is the “ground truth” expression to match.

### 5.3 Lemma B — U-Statistic Unbiasedness for Symmetric Kernels

Use the classical U-statistic fact:

For any symmetric $g(\tau_1,\dots,\tau_K)$ and i.i.d. $\tau_{1:n}$,

$$
\mathbb{E}\left[\frac{1}{\binom{n}{K}}\sum_{|S|=K} g(\tau_S)\right] \;=\; \mathbb{E}[g(\tau_{1:K})].
$$

Apply it with

$$
g(\tau_{1:K}) \;=\; h(\tau_{1:K})\sum_{m=1}^K \nabla_\theta\log\pi_\theta(\tau_m)
$$

to conclude $\mathbb{E}[\widehat{G}_{n,K}]=\nabla_\theta J_{\mathrm{max}@K}$.

### 5.4 Lemma C — Reduce the Subset-Sum Estimator to Rank-Based Weights

Goal: show

$$
\frac{1}{\binom{n}{K}}\sum_{|S|=K} h(\tau_S)\sum_{j\in S}\nabla\log\pi(\tau_j)
\,=\,
\sum_{i=1}^n s_i \nabla\log\pi(\tau_{\sigma(i)}),
$$

and derive explicit $s_i$ by counting the number of $K$-subsets that:
- contain ranked element $i$, and
- have maximum equal to ranked element $j$.

Recommended structure:

1. Expand by swapping sums:
   $$
   \widehat{G}_{n,K}=\sum_{i=1}^n \left(\frac{1}{\binom{n}{K}}\sum_{S\ni i} h(\tau_S)\right)\nabla\log\pi(\tau_i).
   $$
2. Define $s_i := \frac{1}{\binom{n}{K}}\sum_{S\ni i} h(\tau_S)$ and compute it via ranks:
   - When the maximum rank is exactly $j$, the max reward is $R_{\sigma(j)}$.
   - Count subsets with max-rank $j$ that also contain rank $i$ (case split: $j=i$ vs $j>i$).
3. Provide the final closed form for $s_i$ (and check it against special cases).

### 5.5 Optional Lemma D — Marginal Contribution / Baseline Form (RSPO-style)

If the chosen final estimator uses a baseline/marginal contribution decomposition (to avoid “hitchhiking”), include:

- A definition of a per-sample baseline $b_i$ that does **not depend on** $\tau_i$ (or depends only on $\{\tau_\ell:\ell\neq i\}$).
- A proof that replacing $s_i$ with $s_i - b_i c_i$ (for appropriate coefficients $c_i$) preserves unbiasedness.
- A short note deferring *variance* claims to T1.3.

### 5.6 Edge Cases + Robustness Checks (Include in Main Text or Appendix)

- $K=1$: reduces to standard REINFORCE objective and gradient.
- $K=n$: reduces to “leader-only” max gradient.
- Reward ties: show the theorem remains correct with a deterministic tie-break or infinitesimal noise.

---

## 6. Validation Plan (Sanity Checks)

- **Combinatorial sanity**: for small $(n,K)$ (e.g., $(4,2)$, $(5,3)$), explicitly enumerate all $K$-subsets and verify the derived $s_i$ reproduces the subset-sum estimator.
- **Special cases**: verify the $K=1$ and $K=n$ reductions algebraically.
- **Notation consistency**: ensure the final theorem matches the notation/estimator intended for implementation later (T2.1–T2.3).

---

## 7. Work Breakdown (Actionable Steps)

1. Align notation with `docs/Tasks/Task1/task1.1/mathematical_derivation.md` (trajectory, reward, sorting permutation, order statistics).
2. Decide the “canonical” estimator form to prove (subset-sum U-statistic first; then optional baseline/marginal contribution form).
3. Write the theorem statement(s) precisely in `docs/Tasks/Task1/task1.2/unbiasedness_proof.md` (include assumptions).
4. Write Lemma A (score-function gradient identity) with full steps.
5. Write Lemma B (U-statistic unbiasedness) and apply it to the gradient kernel.
6. Write Lemma C (rank-based weight derivation) to obtain implementation-ready $s_i$.
7. Add appendix material (ties, combinatorial identities, enumerative sanity check outline).
8. (Optional) Update `docs/Tasks/Task1/task1.1/mathematical_derivation.md` to link to the detailed proof and remove the sketch.
