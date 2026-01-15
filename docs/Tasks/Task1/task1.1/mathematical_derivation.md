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

### 6.3 Unbiased LOO Gradient Estimator

**Theorem 6.1 (LOO Variance Reduction).** The estimator:

$$
\widehat{\nabla_\theta J}_{\text{max}@K}^{\text{LOO}} = \sum_{i=1}^{n} \left(s_i - b_i^{\text{LOO}} \cdot w_i'\right) \nabla_\theta \log \pi_\theta(\tau_i)
$$

is unbiased, where $w_i'$ is the contribution of sample $i$ to the gradient weight.

**Simplified form:** For the $i$-th ranked sample:

$$
\boxed{\tilde{s}_i = w_i \cdot \left(R_{\sigma(i)} - b_i^{\text{LOO}}\right)}
$$

where:

$$
b_i^{\text{LOO}} = \frac{1}{\binom{n-1}{K}} \sum_{\substack{j=K \\ j \neq i}}^{n} \binom{j'-1}{K-1} R_{\sigma(j)}
$$

Here $j'$ is the rank of sample $j$ in the leave-one-out set.

### 6.4 Computational Efficiency

**Naive computation:** $O(n^2)$ - recompute estimator for each LOO set

**Efficient computation:** $O(n \log n)$ - use incremental updates

```python
def compute_loo_baselines_efficient(sorted_rewards, n, K):
    """
    Compute LOO baselines in O(n) after O(n log n) sorting.
    """
    # Precompute cumulative weighted sums
    total = sum(C(i-1, K-1) * R[i] for i in range(K-1, n)) / C(n, K)

    baselines = []
    for i in range(n):
        if i < K - 1:
            # Sample i doesn't contribute to original estimator
            # LOO estimator uses n-1 samples
            b_i = recompute_with_shifted_ranks(sorted_rewards, i, n-1, K)
        else:
            # Subtract contribution of sample i, adjust normalization
            b_i = adjust_estimator_removing_sample(total, i, n, K)
        baselines.append(b_i)
    return baselines
```

### 6.5 Variance Analysis

**Proposition 6.1.** The LOO baseline reduces variance by eliminating correlation between the sample and its baseline:

$$
\text{Var}[\tilde{s}_i \nabla \log \pi] < \text{Var}[s_i \nabla \log \pi]
$$

when $\text{Cov}(R_i, b_i^{\text{LOO}}) > 0$, which holds in practice.

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

**Definition 7.1 (Hitchhiking).** A sample "hitchhikes" when it receives positive gradient weight despite not contributing to the objective.

**Leader Reward's partial solution:** Only the leader gets boosted, preventing explicit hitchhiking.

**Leader Reward's failure:** It's too aggressive - samples ranked 2nd through $K$-th also deserve credit but receive none.

**Our solution:** Combinatorial weights give each sample credit proportional to its marginal contribution to Max@K, eliminating hitchhiking while preserving proper credit assignment.

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
def maxk_pomo_loss(policy, instance, n_starts, K):
    # POMO: n_starts trajectories from different starting nodes
    trajectories = pomo_rollout(policy, instance, n_starts)
    rewards = [compute_reward(τ) for τ in trajectories]
    log_probs = [policy.log_prob(τ) for τ in trajectories]

    # Our estimator (n = n_starts)
    weights = compute_maxk_weights(rewards, n_starts, K)
    loo_baselines = compute_loo_baselines(rewards, n_starts, K)

    advantages = weights * (rewards - loo_baselines)
    loss = -sum(advantages[i] * log_probs[i] for i in range(n_starts))

    return loss
```

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
| **Theorem 4.1** | Unbiased Max@K gradient with combinatorial weights |
| **Theorem 6.1** | LOO baseline maintains unbiasedness with lower variance |
| **Theorem 7.1** | Leader Reward is biased for Max@K objective |

### 9.2 Key Equations for Implementation

**Weight for $i$-th ranked sample:**
$$
w_i = \frac{\binom{i-1}{K-1}}{\binom{n}{K}} \cdot \mathbf{1}[i \geq K]
$$

**Gradient weight with LOO baseline:**
$$
\tilde{s}_i = w_i \cdot (R_{\sigma(i)} - b_i^{\text{LOO}})
$$

**Policy gradient loss:**
$$
\mathcal{L}(\theta) = -\sum_{i=1}^{n} \tilde{s}_i \log \pi_\theta(\tau_{\sigma(i)})
$$

### 9.3 Computational Complexity

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
