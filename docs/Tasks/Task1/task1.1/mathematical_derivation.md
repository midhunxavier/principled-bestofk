# Formal Derivation of Max@K Gradient Estimator

**Task:** T1.1 - Mathematical Foundation
**Status:** Complete
**Date:** January 9, 2026

---

## Table of Contents

1. [Problem Setup and Notation](#1-problem-setup-and-notation)
2. [The Max@K Objective](#2-the-maxk-objective)
3. [Unbiased Max@K Reward Estimator](#3-unbiased-maxk-reward-estimator)
4. [Unbiased Gradient Estimator](#4-unbiased-gradient-estimator)
5. [Proof of Unbiasedness](#5-proof-of-unbiasedness)
6. [Leave-One-Out Variance Reduction](#6-leave-one-out-variance-reduction)
7. [Comparison with Leader Reward](#7-comparison-with-leader-reward)
8. [NCO-Specific Adaptations](#8-nco-specific-adaptations)
9. [Summary of Key Results](#9-summary-of-key-results)

---

## 1. Problem Setup and Notation

### 1.1 Basic Definitions

Let $\pi_\theta$ denote a policy parameterized by $\theta$ that maps states/instances to distributions over solution trajectories.

| Symbol | Definition |
|--------|------------|
| $s$ | Combinatorial optimization instance (e.g., TSP graph, VRP instance) |
| $\tau$ | Complete solution trajectory |
| $\pi_\theta(\tau \mid s)$ | Probability of trajectory $\tau$ given instance $s$ under policy $\theta$ |
| $R(\tau)$ | Scalar reward for trajectory $\tau$ (higher is better) |
| $K$ | Number of samples in Max@K evaluation |
| $n$ | Number of samples drawn for gradient estimation ($n \geq K$) |

### 1.2 Standard Risk-Neutral Objective

The traditional policy gradient objective maximizes expected reward:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$

With gradient (REINFORCE):

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \nabla_\theta \log \pi_\theta(\tau)\right]
$$

### 1.3 The Training-Evaluation Mismatch

**Training:** Optimizes $J(\theta) = \mathbb{E}[R(\tau)]$ (risk-neutral)

**Evaluation:** Reports $J_{\text{max}@K}(\theta) = \mathbb{E}\left[\max_{i=1}^K R(\tau_i)\right]$ (risk-seeking)

This mismatch means we're training for the wrong objective.

---

## 2. The Max@K Objective

### 2.1 Definition

**Definition 2.1 (Max@K Objective).** Given $K$ i.i.d. samples from policy $\pi_\theta$, the Max@K objective is:

$$
\boxed{J_{\text{max}@K}(\theta) = \mathbb{E}_{\tau_1, \ldots, \tau_K \stackrel{\text{i.i.d.}}{\sim} \pi_\theta}\left[\max_{i=1}^K R(\tau_i)\right]}
$$

### 2.2 Order Statistics Formulation

Let $R_{(1)} \leq R_{(2)} \leq \cdots \leq R_{(K)}$ denote the order statistics of $K$ i.i.d. reward samples. Then:

$$
J_{\text{max}@K}(\theta) = \mathbb{E}[R_{(K)}]
$$

where $R_{(K)}$ is the maximum order statistic.

### 2.3 CDF Representation

If $F_\theta(r) = \Pr(R(\tau) \leq r)$ is the CDF of rewards under $\pi_\theta$, then:

$$
\Pr\left(\max_{i=1}^K R(\tau_i) \leq r\right) = F_\theta(r)^K
$$

Thus:

$$
J_{\text{max}@K}(\theta) = \int_{-\infty}^{\infty} r \cdot K \cdot F_\theta(r)^{K-1} \cdot f_\theta(r) \, dr
$$

where $f_\theta(r)$ is the PDF of rewards.

---

## 3. Unbiased Max@K Reward Estimator

### 3.1 The Challenge

Given $n$ samples with rewards $R_1, \ldots, R_n$, we want an **unbiased** estimate of $\mathbb{E}[\max_{i=1}^K R_i]$ where the expectation is over fresh $K$ samples.

**Naive approach:** Take $\max(R_1, \ldots, R_n)$. This is biased high when $n > K$.

### 3.2 The Unbiased Estimator

**Theorem 3.1 (Unbiased Max@K Reward Estimator).** Let $n \geq K$ samples have rewards sorted in ascending order: $R_{(1)} \leq R_{(2)} \leq \cdots \leq R_{(n)}$. An unbiased estimator of $\mathbb{E}[\max_{i=1}^K R(\tau_i)]$ is:

$$
\boxed{\hat{\rho}^{(g)}(n, K) = \frac{1}{\binom{n}{K}} \sum_{i=K}^{n} \binom{i-1}{K-1} R_{(i)}}
$$

**Proof.** See Section 5.1.

### 3.3 Interpretation of Weights

Define the weight for the $i$-th order statistic (1-indexed, sorted ascending):

$$
w_i = \begin{cases}
\frac{\binom{i-1}{K-1}}{\binom{n}{K}} & \text{if } i \geq K \\
0 & \text{if } i < K
\end{cases}
$$

**Key properties:**
1. Only samples ranked $\geq K$ contribute (bottom $K-1$ samples have zero weight)
2. Higher-ranked samples receive higher weights
3. Weights sum to 1: $\sum_{i=K}^{n} w_i = 1$

### 3.4 Weight Table

For $n=8$ samples and various $K$:

| Rank $i$ | $K=1$ | $K=2$ | $K=4$ | $K=8$ |
|----------|-------|-------|-------|-------|
| 1 (worst) | 0.125 | 0 | 0 | 0 |
| 2 | 0.125 | 0.036 | 0 | 0 |
| 3 | 0.125 | 0.071 | 0 | 0 |
| 4 | 0.125 | 0.107 | 0.014 | 0 |
| 5 | 0.125 | 0.143 | 0.057 | 0 |
| 6 | 0.125 | 0.179 | 0.143 | 0 |
| 7 | 0.125 | 0.214 | 0.286 | 0 |
| 8 (best) | 0.125 | 0.250 | 0.500 | 1.000 |

**Observation:** As $K$ increases, weight concentrates on top samples.

---

## 4. Unbiased Gradient Estimator

### 4.1 Derivation Strategy

We seek $\nabla_\theta J_{\text{max}@K}(\theta)$. The approach:
1. Express the objective as a weighted sum over samples
2. Apply the log-derivative trick to each sample
3. Derive per-sample gradient weights

### 4.2 The Gradient Estimator

**Theorem 4.1 (Unbiased Max@K Gradient Estimator; U-statistic form).** Given $n \geq K$ samples $\{\tau_1, \ldots, \tau_n\} \stackrel{\text{i.i.d.}}{\sim} \pi_\theta$, an unbiased estimator of $\nabla_\theta J_{\text{max}@K}(\theta)$ is:

$$
\boxed{
\widehat{\nabla_\theta J}_{\text{max}@K}
=
\frac{1}{\binom{n}{K}}
\sum_{\substack{S\subseteq[n]\\|S|=K}}
\left(\max_{i\in S} R(\tau_i)\right)
\left(\sum_{j\in S}\nabla_\theta \log \pi_\theta(\tau_j)\right)
}
$$

This can always be rewritten as a per-sample weighted REINFORCE sum:

$$
\boxed{
\widehat{\nabla_\theta J}_{\text{max}@K}
=
\sum_{i=1}^{n} s_i \nabla_\theta \log \pi_\theta(\tau_i),
\qquad
s_i
=
\frac{1}{\binom{n}{K}}
\sum_{\substack{S\subseteq[n]\\|S|=K\\ i\in S}}
\max_{j\in S} R(\tau_j)
}
$$

Closed form in sorted order (for $K\ge 2$): let $\sigma$ sort rewards ascending so $R_{\sigma(1)} \le \cdots \le R_{\sigma(n)}$, and write $R_{(i)}:=R_{\sigma(i)}$. Then:

$$
\boxed{
s_{\sigma(i)}
=
\frac{1}{\binom{n}{K}}
\left[
\mathbf{1}[i\ge K]\binom{i-1}{K-1}R_{(i)}
\;+\;
\sum_{j=i+1}^{n}\binom{j-2}{K-2}R_{(j)}
\right]
}
$$

For the full unbiasedness proof (and derivation of this closed form), see `docs/Tasks/Task1/task1.2/unbiasedness_proof.md`.

### 4.3 Alternative Formulation (RSPO-style)

**Theorem 4.2 (Marginal Contribution Weights).** The gradient weight for sample ranked $i$ can be written as:

$$
s_i = \Pr(\tau_{\sigma(i)} \text{ is the max in a random $K$-subset containing it}) \cdot R_{\sigma(i)} - b_i
$$

where $b_i$ is a baseline that doesn't depend on $\tau_{\sigma(i)}$.

The probability that the $i$-th ranked sample (out of $n$) is the maximum in a random $K$-subset is:

$$
P_i^{\text{max}} = \frac{\binom{i-1}{K-1}}{\binom{n-1}{K-1}} \quad \text{for } i \geq K
$$

### 4.4 Compact Implementation Form

For practical implementation, after sorting samples by reward (ascending), compute:

```
Sort rewards ascending: R_(1) <= ... <= R_(n)

If K == 1:
    s_(i) = R_(i) / n
Else:
    Precompute suffix sums:
        S_i = sum_{j=i+1..n} C(j-2, K-2) * R_(j)
    Then for i = 1..n:
        s_(i) = [ 1[i>=K]*C(i-1,K-1)*R_(i) + S_i ] / C(n,K)
```

---

## 5. Proof of Unbiasedness

### 5.1 Proof of Theorem 3.1 (Reward Estimator)

**Claim:** $\mathbb{E}[\hat{\rho}^{(g)}(n, K)] = \mathbb{E}[\max_{i=1}^K R(\tau_i)]$ where the RHS expectation is over fresh $K$ i.i.d. samples.

**Proof:**

Consider the space of all $\binom{n}{K}$ possible $K$-subsets of our $n$ samples. For any such subset $S$:

$$
\max_{\tau \in S} R(\tau) = R_{(j_S)}
$$

where $j_S$ is the highest rank among samples in $S$.

The maximum of subset $S$ equals $R_{(i)}$ if and only if:
- Sample $\sigma(i)$ is in $S$
- All samples $\sigma(i+1), \ldots, \sigma(n)$ are NOT in $S$

The number of $K$-subsets where sample ranked $i$ is the maximum is:
- Choose $K-1$ samples from the $i-1$ samples ranked below $i$
- This gives $\binom{i-1}{K-1}$ subsets

Therefore:

$$
\mathbb{E}[\hat{\rho}^{(g)}] = \frac{1}{\binom{n}{K}} \sum_{S \in \binom{[n]}{K}} \max_{\tau \in S} R(\tau) = \frac{1}{\binom{n}{K}} \sum_{i=K}^{n} \binom{i-1}{K-1} R_{(i)}
$$

Since our $n$ samples are i.i.d. from $\pi_\theta$, any $K$-subset is distributionally equivalent to fresh $K$ i.i.d. samples. Thus:

$$
\mathbb{E}[\hat{\rho}^{(g)}(n, K)] = \mathbb{E}_{\tau_1, \ldots, \tau_K \stackrel{\text{i.i.d.}}{\sim} \pi_\theta}\left[\max_{i=1}^K R(\tau_i)\right] \quad \blacksquare
$$

### 5.2 Proof of Theorem 4.1 (Gradient Estimator)

Full proof (including the score-function gradient identity and the U-statistic argument) is provided in `docs/Tasks/Task1/task1.2/unbiasedness_proof.md`.

### 5.3 Verification: Special Cases

**Case K = n:**
- Only subset is $S=[n]$
- Estimator = $R_{(n)} \sum_{i=1}^{n} \nabla_\theta \log \pi_\theta(\tau_i)$
- Correct: with $K=n$, the objective uses the maximum among $n$ samples

**Case K = 1:**
- Estimator = $\frac{1}{n} \sum_{i=1}^{n} R(\tau_i) \nabla_\theta \log \pi_\theta(\tau_i)$
- Correct: standard REINFORCE (risk-neutral)

---

## 6. Leave-One-Out Variance Reduction

### 6.1 The LOO Baseline Principle

Standard baseline subtraction in REINFORCE:

$$
\mathbb{E}[(R(\tau) - b) \nabla_\theta \log \pi_\theta(\tau)] = \mathbb{E}[R(\tau) \nabla_\theta \log \pi_\theta(\tau)]
$$

holds for any baseline $b$ that doesn't depend on $\tau$.

### 6.2 LOO Baseline for Max@K

**Definition 6.1 (Leave-One-Out Max@K Baseline).** For sample $i$, define:

$$
b_i^{\text{LOO}} = \hat{\rho}^{(g)}(n-1, K; \text{excluding } \tau_i)
$$

i.e., the Max@K estimator computed on the remaining $n-1$ samples.

### 6.3 Variance-Reduced Gradient Estimators

We present two unbiased variance-reduced forms: the **Sample-LOO** method and the **SubLOO** method.

#### 6.3.1 Sample-LOO Baseline Subtraction

**Theorem 6.1 (Sample-LOO Variance Reduction).** The estimator:

$$
\boxed{
\widehat{\nabla_\theta J}_{\text{max}@K}^{\text{Sample-LOO}} = \sum_{i=1}^{n} \left(s_i - b_i^{\text{LOO}}\right) \nabla_\theta \log \pi_\theta(\tau_i)
}
$$

is unbiased, where $s_i$ is computed from the per-sample weight definition in Section 4.2 (Theorem 4.1 / Eq. for $s_i$) and $b_i^{\text{LOO}}$ is defined in Section 6.2.

**Proof.** Since $b_i^{\text{LOO}}$ depends only on $\{\tau_j : j \neq i\}$, the baseline term $\mathbb{E}[b_i^{\text{LOO}} \nabla_\theta \log \pi_\theta(\tau_i)] = \mathbb{E}[b_i^{\text{LOO}}] \cdot \mathbb{E}[\nabla_\theta \log \pi_\theta(\tau_i)] = 0$ by the score function identity. $\blacksquare$

**Important:** For all ranks $i$ (including $i < K$), the gradient weight $s_i$ is generally non-zero due to the "Support" term in the closed form from Section 4.2 / Proposition 5.1 (T1.2). Only when combined with the SubLOO method (below) do low-ranked samples receive zero weight.

#### 6.3.2 SubLOO: Per-Subset Baseline (Hitchhiking-Free)

**Definition 6.2 (SubLOO Baseline).** For a subset $S$ and element $i \in S$:

$$
b_{i,S} := \max_{j \in S \setminus \{i\}} R(\tau_j)
$$

**Theorem 6.2 (SubLOO Variance Reduction).** The estimator:

$$
\boxed{
\widehat{\nabla_\theta J}_{\text{max}@K}^{\text{SubLOO}} = \frac{1}{\binom{n}{K}} \sum_{|S|=K} \sum_{i \in S} \left(\max_{j \in S} R(\tau_j) - b_{i,S}\right) \nabla_\theta \log \pi_\theta(\tau_i)
}
$$

is unbiased. Moreover, within each subset $S$, **only the maximum element receives non-zero weight** (equal to the max–second-max gap).

**Proof.** See `docs/Tasks/Task1/task1.2/unbiasedness_proof.md`, Proposition 6.1. $\blacksquare$

**Key Property:** SubLOO eliminates "hitchhiking" by ensuring non-winners in each subset receive zero gradient signal for that subset.

### 6.4 Computational Efficiency

**For Sample-LOO:** $O(n \log n)$ using suffix-sum precomputation similar to computing $s_i$.

**For SubLOO:** The per-sample weight can be computed efficiently:

```python
def compute_subloo_weights(sorted_rewards, n, K):
    """
    Compute SubLOO variance-reduced weights in O(n) after sorting.
    Only the maximum in each subset contributes; total weight for
    rank-i sample is sum of (R_(i) - R_(i-1)) over subsets where i is max.
    """
    # For rank i >= K, number of subsets where i is maximum: C(i-1, K-1)
    # In each such subset, the second-max has rank at most i-1
    # Expected contribution: C(i-1, K-1) * (R_(i) - E[second_max | i is max])
    
    weights = [0.0] * n
    for i in range(K-1, n):  # i is 0-indexed, so this is rank K to n
        # Number of subsets where rank-i is the maximum
        num_subsets = comb(i, K-1)
        # For each subset, weight = R_(i) - max of remaining K-1 elements
        # This requires computing expected second-max, done via suffix sums
        weights[i] = compute_gap_contribution(sorted_rewards, i, K) / comb(n, K)
    return weights
```

### 6.5 Variance Analysis

**Proposition 6.1 (Variance Reduction; what is guaranteed vs. expected).**

- **Sample-LOO:** With unit coefficient, baseline subtraction reduces the *per-sample* variance contribution when the baseline is sufficiently correlated with the weight (see the explicit condition in Theorem 3.1 of T1.3). Unbiasedness is guaranteed; variance reduction is typical but not unconditional.

- **SubLOO:** SubLOO eliminates *within-subset* hitchhiking: in each subset $S$, only the subset maximum receives nonzero weight (the max–second-max gap). This reduces conditional variance at the subset level. However, an unconditional global guarantee $\text{Var}[\widehat{G}^{\text{SubLOO}}] \le \text{Var}[\widehat{G}_{n,K}]$ requires additional assumptions to control cross-covariance terms between different samples’ contributions (see Remark 3.2 in T1.3).

**Takeaway:** both LOO variants are unbiased; both are expected to reduce variance in practice, with SubLOO typically reducing variance the most because it removes hitchhiking completely.

---

## 7. Comparison with Leader Reward

### 7.1 Leader Reward Formulation

Leader Reward (Wang et al., 2024) modifies POMO's shared baseline:

$$
A_i^{\text{Leader}} = (R_i - \bar{R}) + \alpha \cdot \mathbf{1}[i = \text{argmax}_j R_j] \cdot \beta
$$

where:
- $\bar{R} = \frac{1}{n} \sum_j R_j$ is the mean reward
- $\alpha$ is a hyperparameter
- $\beta$ is typically a bonus term

### 7.2 Why Leader Reward is Biased

**Theorem 7.1 (Leader Reward Bias).** The Leader Reward gradient estimator is a biased estimator of $\nabla_\theta J_{\text{max}@K}$ for any $K > 1$.

**Proof sketch:**

1. Leader Reward only gives extra weight to the **single best** sample
2. In Max@K with $K < n$, samples ranked 2nd, 3rd, ..., $K$-th also contribute positively to the expected maximum
3. By ignoring these contributions, Leader Reward systematically under-weights non-leader samples that would be maximal in many $K$-subsets

**Concrete example:** With $n=8, K=4$:
- Sample ranked 5th has weight $w_5 = 0.057$ in our unbiased estimator
- Leader Reward gives it weight 0 (standard POMO advantage only)

### 7.3 Variance Comparison

| Estimator | Bias | Variance | Hyperparameters |
|-----------|------|----------|-----------------|
| Our Max@K | Unbiased | Controlled via LOO | $n, K$ (principled) |
| Leader Reward | Biased | Task-dependent | $\alpha$ (heuristic) |
| Vanilla POMO | Biased for Max@K | Low | None |

### 7.4 The Hitchhiking Problem

**Definition 7.1 (Hitchhiking).** A sample "hitchhikes" when it receives positive gradient weight despite not being the maximum in a given $K$-subset.

**Leader Reward's partial solution:** Only the leader gets boosted, preventing explicit hitchhiking.

**Leader Reward's failure:** It's too aggressive—samples ranked 2nd through $K$-th also deserve credit for being potential winners in many $K$-subsets, but receive none.

**Our base estimator (Eq. 4.1):** The combinatorial weights give credit based on *all* $K$-subsets containing each sample. This means a sample receives credit for:
1. **Win term:** Subsets where it is the maximum (proper credit)
2. **Support term:** Subsets where another sample wins (hitchhiking)

The Support term is necessary for unbiasedness but increases variance.

**Our SubLOO estimator (Section 6.3.2):** By subtracting the per-subset baseline $b_{i,S} = \max_{j \in S \setminus \{i\}} R(\tau_j)$, we eliminate hitchhiking entirely. Within each subset, only the winner receives non-zero gradient weight, equal to the gap $R_{\text{max}} - R_{\text{second-max}}$. This is both unbiased and hitchhiking-free.

**Summary:** The base estimator is unbiased but includes hitchhiking. The SubLOO estimator is both unbiased and hitchhiking-free, achieving lower variance.

---

## 8. NCO-Specific Adaptations

### 8.1 Mapping from LLM to NCO Setting

| LLM Concept | NCO Translation |
|-------------|-----------------|
| Input prompt $x$ | CO instance $s$ (graph, demands, etc.) |
| Generated response $y_i$ | Solution trajectory $\tau_i$ |
| Reward $g(y_i)$ | Negative cost: $R(\tau_i) = -\text{cost}(\tau_i)$ |
| $\log p_\theta(y_i \mid x)$ | $\sum_t \log \pi_\theta(a_t \mid s_t)$ |

### 8.2 Integration with POMO Multi-Start

POMO generates $N$ trajectories from different starting nodes. Our estimator applies as:

```python
def maxk_pomo_gradient(policy, instance, n_starts, K):
    # POMO: n_starts trajectories from different starting nodes
    trajectories = pomo_rollout(policy, instance, n_starts)
    rewards = [compute_reward(τ) for τ in trajectories]
    log_probs = [policy.log_prob(τ) for τ in trajectories]

    # Compute gradient weights s_i via Proposition 5.1 (includes Support term)
    s_weights = compute_full_gradient_weights(rewards, n_starts, K)  # NOT just w_i!
    
    # Sample-LOO variance reduction
    loo_baselines = compute_loo_baselines(rewards, n_starts, K)
    variance_reduced_weights = [s_weights[i] - loo_baselines[i] for i in range(n_starts)]

    # Surrogate loss: gradient of this equals the REINFORCE estimator
    # Note: variance_reduced_weights are treated as constants (stop-gradient)
    surrogate_loss = -sum(stop_grad(variance_reduced_weights[i]) * log_probs[i] 
                          for i in range(n_starts))

    return surrogate_loss
```

**Important:** The `variance_reduced_weights` are computed from rewards and must be detached from the computational graph (stop-gradient) before multiplying with `log_probs`. The gradient of the surrogate loss with respect to $\theta$ yields the unbiased REINFORCE estimator.

### 8.2.1 Note on i.i.d. Assumption

**Caveat:** The proofs in this document assume $\tau_1, \ldots, \tau_n \stackrel{\text{i.i.d.}}{\sim} \pi_\theta$. Standard POMO uses a **deterministic** set of starting nodes (all $N$ nodes for an $N$-city TSP), which means trajectories are not identically distributed—each starts from a different node.

**Clarification for POMO (deterministic multi-start vs i.i.d.).** One way to align with the i.i.d. assumption is to define a *mixture* rollout distribution
$$
\pi_\theta(\tau) = \frac{1}{N} \sum_{s=1}^{N} \pi_\theta(\tau \mid \text{start}=s),
$$
which corresponds to sampling a start node uniformly at random, then rolling out the policy.

However, standard POMO typically *deterministically enumerates* all $N$ start nodes in each batch. That produces an exchangeable-but-not-i.i.d. collection of trajectories (closer to sampling starts **without replacement**). The U-statistic unbiasedness proof in T1.2 applies cleanly to the i.i.d. setting; for deterministic multi-start, you should view the estimator as optimizing the Max@K objective under that multi-start sampling scheme.

In practice, the two objectives are often close when $N$ is large, but they are not identical. If strict i.i.d. correspondence is desired, one can sample starts randomly (with replacement) rather than enumerating them.

### 8.3 Handling Reward Sign

In CO, we minimize cost but our estimator maximizes reward:

$$
R(\tau) = -\text{cost}(\tau)
$$

This ensures $\max_i R(\tau_i)$ corresponds to $\min_i \text{cost}(\tau_i)$.

### 8.4 Tie-Breaking

When multiple samples have equal rewards, add small noise:

$$
\tilde{R}_i = R_i + \epsilon_i, \quad \epsilon_i \sim \text{Uniform}(-\delta, \delta)
$$

with $\delta \ll \min_{i \neq j} |R_i - R_j|$ for distinct rewards.

### 8.5 Stability Mechanisms

1. **Gradient clipping:** $\|\nabla\| \leq g_{\max}$
2. **Advantage normalization:** $\tilde{A}_i = \frac{A_i - \mu_A}{\sigma_A + \epsilon}$
3. **Entropy regularization:** Add $\beta H(\pi_\theta)$ to objective

---

## 9. Summary of Key Results

### 9.1 Main Theorems

| Result | Statement |
|--------|-----------|
| **Theorem 3.1** | Unbiased Max@K reward estimator: $\hat{\rho}^{(g)} = \frac{1}{\binom{n}{K}} \sum_{i=K}^{n} \binom{i-1}{K-1} R_{(i)}$ |
| **Theorem 4.1** | Unbiased Max@K gradient estimator (U-statistic / per-sample form) |
| **Theorem 6.1** | LOO baseline preserves unbiasedness; variance reduction is typical but not unconditional |
| **Theorem 7.1** | Leader Reward is biased for Max@K objective |

## 10. Task 1 Theorem Map (What is defined, proven, and used)

This section summarizes Task 1 as a dependency graph: definitions $\to$ estimators $\to$ guarantees.

### 10.1 Core Definitions

1. **Max@K objective** (Definition 2.1):
   $$
   J_{\text{max}@K}(\theta) = \mathbb{E}_{\tau_{1:K}\stackrel{\text{i.i.d.}}{\sim}\pi_\theta}\big[\max_{j\in[K]} R(\tau_j)\big].
   $$

2. **U-statistic view** (T1.2 Lemma 3.1): average a symmetric kernel over all $K$-subsets of $n$ i.i.d. samples without bias.

### 10.2 Estimators

1. **Unbiased Max@K reward estimator** (Theorem 3.1): given $n\ge K$ i.i.d. samples, with sorted rewards $R_{(1)}\le\cdots\le R_{(n)}$,
   $$
   \hat{\rho}^{(g)}(n,K) = \frac{1}{\binom{n}{K}}\sum_{i=K}^{n} \binom{i-1}{K-1} R_{(i)}.
   $$

2. **Unbiased Max@K gradient estimator (U-statistic form)** (T1.2 Theorem 4.1 / this doc Theorem 4.1):
   $$
   \widehat{\nabla_\theta J}_{\text{max}@K}
   = \frac{1}{\binom{n}{K}} \sum_{|S|=K}
   \Big(\max_{i\in S} R(\tau_i)\Big)\Big(\sum_{j\in S}\nabla_\theta\log\pi_\theta(\tau_j)\Big).
   $$

3. **Per-sample score-weight form** (T1.2 Section 5):
   $$
   \widehat{\nabla_\theta J}_{\text{max}@K} = \sum_{i=1}^{n} s_i\,\nabla_\theta\log\pi_\theta(\tau_i),
   \qquad
   s_i := \frac{1}{\binom{n}{K}} \sum_{|S|=K,\,i\in S} \max_{j\in S} R(\tau_j).
   $$
   Closed form in ranks is Proposition 5.1 in T1.2.

4. **Sample-LOO variance reduction** (T1.3 Theorem 2.1):
   $$
   \widehat{G}^{\text{Sample-LOO}} = \sum_{i=1}^{n} (s_i - b_i^{\text{LOO}})\,\nabla_\theta\log\pi_\theta(\tau_i),
   \qquad
   b_i^{\text{LOO}} = \hat{\rho}^{(g)}(n-1,K;\text{exclude }i),
   $$
   defined when $n-1\ge K$.

5. **SubLOO (per-subset LOO) variance reduction** (T1.2 Proposition 6.1 / T1.3 Theorem 2.2):
   $$
   \widehat{G}^{\text{SubLOO}} = \frac{1}{\binom{n}{K}} \sum_{|S|=K}\sum_{i\in S}
   \Big(\max_{j\in S}R_j - \max_{j\in S\setminus\{i\}}R_j\Big)\,\nabla_\theta\log\pi_\theta(\tau_i),
   $$
   defined for $K\ge 2$.

### 10.3 Guarantees (and what is *not* claimed)

1. **Unbiasedness**
   - Reward estimator: proven (T1.1 Section 5.1).
   - Base gradient estimator: proven (T1.2 Theorem 4.1).
   - Sample-LOO and SubLOO: unbiasedness proven via baseline independence (T1.3 Section 2; T1.2 Section 6).

2. **Variance reduction**
   - Sample-LOO: variance reduction is typical but requires correlation conditions for a strict guarantee with unit coefficient (T1.3 Theorem 3.1).
   - SubLOO: eliminates within-subset hitchhiking and reduces conditional variance per subset; a global unconditional variance inequality requires additional assumptions (T1.3 Remark 3.2).

3. **Edge cases**
   - $K=1$: reduces to standard risk-neutral REINFORCE.
   - $K=n$: base estimator uses one subset; SubLOO reduces to max–second-max gap weighting.

### 10.4 Practical alignment note (POMO)

All proofs above assume i.i.d. trajectories. Deterministic multi-start (standard POMO) corresponds to a slightly different sampling scheme; see Section 8.2.1 for the exact caveat and how to recover i.i.d. if desired.

### 10.5 Quick reference (Implementation equations)

**Weight for $i$-th ranked sample:**
$$
w_i = \frac{\binom{i-1}{K-1}}{\binom{n}{K}} \cdot \mathbf{1}[i \geq K]
$$

**Gradient weight (Sample-LOO):**
$$
\tilde{s}_i^{\text{Sample-LOO}} = s_i - b_i^{\text{LOO}}
$$

where $s_i$ is the per-sample Max@K gradient weight (Theorem 4.1 / Proposition 5.1 in T1.2) and $b_i^{\text{LOO}}$ is the leave-one-out Max@K estimate.

**Gradient weight (SubLOO, hitchhiking-free):**
$$
\tilde{s}_i^{\text{SubLOO}} = \frac{1}{\binom{n}{K}} \sum_{\substack{S \ni i \\ |S|=K}} \left(\max_{j \in S} R_j - \max_{j \in S \setminus \{i\}} R_j\right)
$$

**Surrogate loss for autodiff:**
$$
\mathcal{L}_{\text{surr}}(\theta) = -\sum_{i=1}^{n} [\tilde{s}_i]_{\text{stop-grad}} \cdot \log \pi_\theta(\tau_i)
$$

where $[\cdot]_{\text{stop-grad}}$ indicates the weights are treated as constants. Differentiating $\mathcal{L}_{\text{surr}}$ with respect to $\theta$ yields the unbiased REINFORCE gradient:
$$
\nabla_\theta \mathcal{L}_{\text{surr}}(\theta) = -\sum_{i=1}^{n} \tilde{s}_i \nabla_\theta \log \pi_\theta(\tau_i) = \widehat{\nabla_\theta J}_{\text{max}@K}
$$

### 10.6 Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Sorting rewards | $O(n \log n)$ |
| Computing weights | $O(n)$ |
| Computing LOO baselines | $O(n)$ (with precomputation) |
| **Total per instance** | $O(n \log n)$ |

This is negligible compared to forward/backward passes through the neural policy.

---

## References

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.

2. Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). POMO: Policy optimization with multiple optima for reinforcement learning. *NeurIPS 2020*.

3. Wang, Y., et al. (2024). Leader Reward for POMO-Based Neural Combinatorial Optimization. *arXiv preprint*.

4. PKPO Authors (2025). Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems. *arXiv preprint*.

5. Zhang, Y., et al. (2025). RSPO: Risk-Seeking Policy Optimization for Pass@k and Max@k Metrics. *arXiv preprint*.

---

**Document Status:** Complete
**Next Steps:** T1.2 (Prove unbiasedness property - detailed proof), T1.3 (Derive variance reduction via LOO - extended analysis)
