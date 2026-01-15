# Prove Unbiasedness Property (T1.2)

**Task:** T1.2 — Prove unbiasedness property  
**Status:** Draft (theorem + full proof)  
**Date:** January 15, 2026  

---

## 1. Setup and Assumptions

Fix a single combinatorial optimization instance $s$ and suppress conditioning on $s$ to simplify notation.

- Let $\pi_\theta(\tau)$ denote a policy (probability mass function or density) over trajectories $\tau \in \mathcal{T}$, parameterized by $\theta$.
- Let $R(\tau) \in \mathbb{R}$ be the (scalar) reward, where larger is better.
- For $K \in \mathbb{N}$, define the Max@K objective

$$
J_{\mathrm{max}@K}(\theta)
\;=\;
\mathbb{E}_{\tau_{1:K}\stackrel{\mathrm{i.i.d.}}{\sim}\pi_\theta}\!\left[\max_{j\in[K]} R(\tau_j)\right].
$$

**Regularity assumptions (standard REINFORCE conditions).** Assume:

1. $\pi_\theta(\tau)$ is differentiable in $\theta$ and $\nabla_\theta \log \pi_\theta(\tau)$ exists $\pi_\theta$-a.s.
2. The exchange of gradient and expectation is justified, e.g. there exists an integrable dominating function so that
   $$
   \nabla_\theta \mathbb{E}[f(\tau)] = \mathbb{E}[f(\tau)\nabla_\theta\log\pi_\theta(\tau)]
   $$
   holds for the functions used below.
3. Integrability: $\mathbb{E}[|R(\tau)|] < \infty$ and the score-weighted terms have finite expectation (enough for all steps below).

**Tie handling.** If rewards can tie with nonzero probability, fix any deterministic tie-break rule (e.g. break ties by sample index) or use an infinitesimal continuous perturbation; the unbiasedness arguments below do not depend on the particular tie-break.

---

## 2. Lemma: Score-Function Gradient of the Max@K Objective

Define the symmetric kernel

$$
h(\tau_{1:K}) := \max_{j\in[K]} R(\tau_j).
$$

**Lemma 2.1 (Score-function form).**

$$
\nabla_\theta J_{\mathrm{max}@K}(\theta)
\;=\;
\mathbb{E}_{\tau_{1:K}\stackrel{\mathrm{i.i.d.}}{\sim}\pi_\theta}\!\left[
h(\tau_{1:K}) \sum_{j=1}^{K} \nabla_\theta \log \pi_\theta(\tau_j)
\right].
$$

**Proof.**
Write the expectation as an integral (or sum) over the product distribution:

$$
J_{\mathrm{max}@K}(\theta)=\int h(\tau_{1:K}) \prod_{j=1}^K \pi_\theta(\tau_j)\, d\tau_{1:K}.
$$

Differentiate under the integral sign (Assumption 2) and use $\nabla_\theta \pi_\theta(\tau)=\pi_\theta(\tau)\nabla_\theta\log\pi_\theta(\tau)$:

$$
\begin{aligned}
\nabla_\theta J_{\mathrm{max}@K}(\theta)
&= \int h(\tau_{1:K}) \nabla_\theta \left(\prod_{j=1}^K \pi_\theta(\tau_j)\right)\, d\tau_{1:K} \\
&= \int h(\tau_{1:K}) \left(\prod_{j=1}^K \pi_\theta(\tau_j)\right)\left(\sum_{j=1}^K \nabla_\theta \log\pi_\theta(\tau_j)\right)\, d\tau_{1:K} \\
&= \mathbb{E}_{\tau_{1:K}}\!\left[h(\tau_{1:K}) \sum_{j=1}^K \nabla_\theta \log\pi_\theta(\tau_j)\right]. \quad \blacksquare
\end{aligned}
$$

---

## 3. Lemma: U-Statistic Unbiasedness (Subset Averaging)

Let $\tau_1,\dots,\tau_n \stackrel{\mathrm{i.i.d.}}{\sim} \pi_\theta$ with $n \ge K$.
For any $K$-subset $S\subseteq[n]$ with $|S|=K$, denote $\tau_S = (\tau_i)_{i\in S}$.

**Lemma 3.1 (U-statistic expectation).**
If $g:\mathcal{T}^K\to\mathbb{R}^d$ is symmetric and integrable, then

$$
U_{n,K}
:=
\frac{1}{\binom{n}{K}} \sum_{\substack{S\subseteq[n]\\ |S|=K}} g(\tau_S)
\quad\Rightarrow\quad
\mathbb{E}[U_{n,K}] = \mathbb{E}[g(\tau_{1:K})].
$$

**Proof.**
Each term $g(\tau_S)$ has the same distribution as $g(\tau_{1:K})$ because $(\tau_1,\dots,\tau_n)$ are i.i.d. and $g$ is symmetric in its arguments. Therefore,

$$
\mathbb{E}[U_{n,K}]
=
\frac{1}{\binom{n}{K}} \sum_{|S|=K} \mathbb{E}[g(\tau_S)]
=
\frac{1}{\binom{n}{K}} \sum_{|S|=K} \mathbb{E}[g(\tau_{1:K})]
=
\mathbb{E}[g(\tau_{1:K})]. \quad \blacksquare
$$

---

## 4. Theorem: Unbiased Max@K Gradient Estimator (U-Statistic Form)

Define the $K$-sample gradient kernel

$$
g(\tau_{1:K})
:=
h(\tau_{1:K})\sum_{j=1}^K \nabla_\theta \log \pi_\theta(\tau_j).
$$

Given $n\ge K$ i.i.d. samples $\tau_{1:n}$, define the estimator

$$
\widehat{G}_{n,K}(\theta)
:=
\frac{1}{\binom{n}{K}}
\sum_{\substack{S\subseteq[n]\\|S|=K}}
\left(
\max_{i\in S} R(\tau_i)
\right)
\left(
\sum_{j\in S}\nabla_\theta \log\pi_\theta(\tau_j)
\right).
$$

**Theorem 4.1 (Unbiasedness).**

$$
\mathbb{E}\big[\widehat{G}_{n,K}(\theta)\big]
\;=\;
\nabla_\theta J_{\mathrm{max}@K}(\theta).
$$

**Proof.**
By Lemma 2.1, $\nabla_\theta J_{\mathrm{max}@K}(\theta) = \mathbb{E}[g(\tau_{1:K})]$.
By Lemma 3.1 (U-statistic expectation), $\mathbb{E}[\widehat{G}_{n,K}(\theta)] = \mathbb{E}[g(\tau_{1:K})]$.
Combining yields $\mathbb{E}[\widehat{G}_{n,K}(\theta)] = \nabla_\theta J_{\mathrm{max}@K}(\theta)$. $\blacksquare$

---

## 5. Implementation Form: Per-Sample Score Weights

The U-statistic form in Theorem 4.1 can always be rewritten as a per-sample weighted REINFORCE sum.

Let

$$
\psi_i
:=
\nabla_\theta \log \pi_\theta(\tau_i).
$$

Swap the subset sum and the per-subset score sum:

$$
\widehat{G}_{n,K}(\theta)
=
\sum_{i=1}^{n} s_i \, \psi_i,
\qquad
s_i
:=
\frac{1}{\binom{n}{K}}
\sum_{\substack{S\subseteq[n]\\|S|=K\\ i\in S}}
\max_{j\in S} R(\tau_j).
$$

So $s_i$ is the average Max@K reward (over all $K$-subsets) conditional on the subset containing sample $i$.

### 5.1 Closed Form via Order Statistics

Let $\sigma$ be the permutation that sorts rewards ascending:

$$
R_{\sigma(1)} \le \cdots \le R_{\sigma(n)},
\qquad
R_{(i)} := R_{\sigma(i)},\;\;\tau_{(i)}:=\tau_{\sigma(i)}.
$$

Assume $K\ge 2$ (the $K=1$ case is given in Section 5.2).

**Proposition 5.1 (Closed form for $s_{\sigma(i)}$).**
For rank $i\in[n]$,

$$
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
$$

**Proof.**
Fix a rank $i$ and consider all $K$-subsets $S$ that contain the element at rank $i$.
For such a subset, the maximum rank in $S$ is some $j \in \{i,\dots,n\}$, and the subset’s max reward is $R_{(j)}$.

- **Case 1: the max rank equals $j=i$.** Then all other $K-1$ elements must be chosen from ranks $\{1,\dots,i-1\}$, giving $\binom{i-1}{K-1}$ subsets (and this requires $i\ge K$).
- **Case 2: the max rank equals $j>i$.** Then the subset must include ranks $i$ and $j$, and the remaining $K-2$ elements are chosen from ranks $\{1,\dots,j-1\}\setminus\{i\}$, which has size $j-2$. This gives $\binom{j-2}{K-2}$ subsets.

Summing the max reward $R_{(j)}$ over these disjoint cases and dividing by $\binom{n}{K}$ yields the stated formula. $\blacksquare$

### 5.2 Special Case: $K=1$

For $K=1$, the objective is risk-neutral:

$$
J_{\mathrm{max}@1}(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)].
$$

The U-statistic estimator reduces to standard REINFORCE:

$$
\widehat{G}_{n,1}(\theta)
=
\frac{1}{n}\sum_{i=1}^n R(\tau_i)\nabla_\theta\log\pi_\theta(\tau_i).
$$

### 5.3 Special Case: $K=n$

When $K=n$, there is only one subset $S=[n]$:

$$
\widehat{G}_{n,n}(\theta)=\left(\max_{i\in[n]}R(\tau_i)\right)\left(\sum_{j=1}^n \nabla_\theta\log\pi_\theta(\tau_j)\right).
$$

In this case, $s_i = \max_{j\in[n]}R(\tau_j)$ for all $i$, consistent with Proposition 5.1.

---

## 6. (Optional) Hitchhiking-Free Variant via Per-Subset Leave-One-Out Baselines

The estimator in Theorem 4.1 is unbiased but can exhibit “hitchhiking” because non-maximum samples in a subset receive the same max reward signal.
A standard control-variate idea is to subtract, for each score term, a baseline that depends only on the *other* samples in the same subset.

For a fixed subset $S$ and element $i\in S$, define the per-subset leave-one-out baseline

$$
b_{i,S} := \max_{j\in S\setminus\{i\}} R(\tau_j),
$$

with the convention that $\max(\varnothing)=-\infty$ so that when $K=1$ the baseline is irrelevant.

Define the modified estimator:

$$
\widehat{G}^{\mathrm{subLOO}}_{n,K}(\theta)
:=
\frac{1}{\binom{n}{K}}
\sum_{|S|=K}\;
\sum_{i\in S}
\left(
\max_{j\in S}R(\tau_j) - b_{i,S}
\right)
\nabla_\theta\log\pi_\theta(\tau_i).
$$

**Proposition 6.1 (Unbiasedness preserved).**

$$
\mathbb{E}\big[\widehat{G}^{\mathrm{subLOO}}_{n,K}(\theta)\big]
=
\mathbb{E}\big[\widehat{G}_{n,K}(\theta)\big]
=
\nabla_\theta J_{\mathrm{max}@K}(\theta).
$$

**Proof.**
Fix $S$ and $i\in S$. Conditional on $\{\tau_j:j\in S\setminus\{i\}\}$, the baseline $b_{i,S}$ is a constant and $\tau_i$ is independent of those other samples. Therefore,

$$
\mathbb{E}\!\left[b_{i,S}\nabla_\theta\log\pi_\theta(\tau_i)\;\middle|\;\{\tau_j:j\in S\setminus\{i\}\}\right]
=
b_{i,S}\,\mathbb{E}\!\left[\nabla_\theta\log\pi_\theta(\tau_i)\right]
=
0.
$$

Taking expectation again shows each subtracted baseline term has zero mean, so $\mathbb{E}[\widehat{G}^{\mathrm{subLOO}}_{n,K}]=\mathbb{E}[\widehat{G}_{n,K}]$. $\blacksquare$

**Interpretation (no hitchhiking within a subset).**
For any fixed subset $S$, the quantity $\max_{j\in S}R(\tau_j)-b_{i,S}$ equals $0$ for every non-maximum element, and equals the max–second-max gap for the maximum element. Thus only the subset’s leader receives nonzero weight.

---

## Appendix A. Small $(n,K)$ Enumeration Sanity Check (Outline)

For $(n,K)=(4,2)$, enumerate all $\binom{4}{2}=6$ subsets $S$ and verify:

1. The U-statistic form in Theorem 4.1 equals $\sum_i s_i \psi_i$ with $s_i$ defined as the “subset-average max” for subsets containing $i$.
2. Proposition 5.1 matches the explicit enumeration counts:
   - max-rank $j=i$ contributes $\binom{i-1}{1}$ subsets,
   - max-rank $j>i$ contributes $\binom{j-2}{0}=1$ subset per $j$.

This check is purely combinatorial and independent of $\pi_\theta$.

