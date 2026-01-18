# Product Requirements Document (PRD)
## Principled Max@K / Best-of-K RL for Combinatorial Optimization

**Document Version:** 1.0  
**Created:** January 9, 2026  
**Status:** Draft  

---

## 1. Executive Summary

### 1.1 Problem Statement
Current approaches to training neural combinatorial optimization (NCO) models suffer from a fundamental mismatch: **models are trained to optimize expected reward (risk-neutral) but evaluated on the best-of-K samples (risk-seeking)**. While heuristic solutions like Leader Reward exist, they lack principled mathematical foundations and do not provide unbiased gradient estimates for the Max@K objective.

### 1.2 Proposed Solution
Develop an **unbiased and variance-reduced gradient estimator** for Max@K / Best-of-K objectives in RL4CO (Reinforcement Learning for Combinatorial Optimization). This principled approach will directly optimize the evaluation metric used in practice: obtaining the best solution among K sampled candidates.

### 1.3 Key Value Propositions
- **Mathematically principled**: Derived unbiased estimator for ∇J_maxK vs. heuristic Leader Reward
- **Lower variance**: Variance reduction techniques ensure faster, more stable convergence
- **Generalizability**: Applicable across multiple RL4CO benchmark tasks (TSP, VRP, OP, PCTSP, etc.)
- **Publication-ready**: Novel contribution bridging RSPO/PKPO ideas (from LLM settings) to combinatorial optimization

---

## 2. Background & Motivation

### 2.1 Current State of the Art

#### 2.1.1 Risk-Neutral Training Paradigm
Existing NCO methods (Bello 2017, Kool 2019, POMO 2020) train policies to optimize:

```
J(θ) = E_{τ~π_θ}[ R(τ) ]
```

But at evaluation time, practitioners sample K solutions and take the best:

```
J_maxK(θ) = E_{τ1..τK i.i.d.~π_θ}[ max_i R(τi) ]
```

This mismatch leaves performance on the table.

#### 2.1.2 Leader Reward (Wang et al., 2024)
- **Approach**: Heuristically boost the gradient weight of the "leader" (best) trajectory
- **Formula (informal)**: `A_i = (R_i − mean_R) + α * I[i is leader] * (bonus)`
- **Pros**: Simple, nearly zero overhead, often works empirically
- **Cons**:
  - Not derived as unbiased estimator of ∇J_maxK
  - Heuristic credit assignment (only leader gets boost)
  - Task-dependent hyperparameter α
  - Can destabilize training at large α values

#### 2.1.3 PKPO & RSPO (2025)
Recent work in LLM post-training provides the mathematical foundation we can adapt:
- **PKPO**: Unbiased estimators for pass@k and maxg@k with LOO variance reduction
- **RSPO**: Marginal contribution framing that fixes "hitchhiking" problem

**The Gap**: These methods target LLM settings and haven't been adapted to structured autoregressive NCO rollouts with POMO-style shared baselines.

### 2.2 The Hitchhiking Problem
When drawing K samples and using max reward as a shared signal:
> Garbage samples that co-occurred with one good sample accidentally get reinforced, creating biased gradients and often slowing learning.

Our principled estimator must eliminate this hitchhiking effect.

---

## 3. Goals & Objectives

### 3.1 Primary Goals

| Goal ID | Description | Success Metric |
|---------|-------------|----------------|
| G1 | Derive clean, unbiased gradient estimator for Max@K | Mathematical proof of unbiasedness |
| G2 | Demonstrate lower variance than Leader Reward | Empirical variance measurements |
| G3 | Show faster convergence than Leader Reward | Training curves comparison |
| G4 | Generalize across multiple RL4CO tasks | Performance on TSP, CVRP, OP, PCTSP |

### 3.2 Secondary Goals

| Goal ID | Description | Success Metric |
|---------|-------------|----------------|
| G5 | Provide drop-in integration with RL4CO library | Pull-request ready code |
| G6 | Develop stability recipes for NCO | Documented best practices |
| G7 | Comprehensive ablation studies | Clear understanding of design choices |

### 3.3 Non-Goals (Out of Scope)
- Developing new policy architectures
- Optimizing inference-time search
- Extending to multi-objective optimization (for now)

---

## 4. Technical Approach

### 4.1 Core Mathematical Framework

#### 4.1.1 Max@K Objective
```
J_maxK(θ) = E_{τ1..τK i.i.d.~π_θ}[ max_i R(τi) ]
```

#### 4.1.2 Unbiased Reward + Gradient Estimators (from PKPO/RSPO)

Given `n ≥ k` sampled trajectories per instance, let rewards be sorted ascending:
`R_(1) ≤ R_(2) ≤ ... ≤ R_(n)` (ties broken deterministically or with tiny noise).

**Unbiased estimator of Max@k reward (Task 1.1):**
```
ρ^(g)(n,k) = Σ_{i=1..n} w_i * R_(i)

w_i = 1[i ≥ k] * C(i−1, k−1) / C(n,k)
```

**Unbiased gradient estimator (Task 1.2):**
```
∇̂J_maxK = Σ_{i=1..n} s_i * ∇_θ log π_θ(τ_(i))
```

Important: the gradient score-weights `s_i` are *not* just `w_i * R_(i)`.
For `k ≥ 2`, `s_i` includes a “Support” term accounting for subsets where a
higher-ranked sample is the max (see `docs/Tasks/Task1/task1.2/unbiasedness_proof.md`).

#### 4.1.3 Variance Reduction via LOO Baselines
- Leave-one-out baselines that maintain unbiasedness
- **Sample-LOO:** per-sample baseline from the other `n-1` rewards (requires `n > k`)
- **SubLOO:** per-subset baseline (max–second-max gap; requires `k ≥ 2`)
- Compatible with POMO-style shared baseline structure

### 4.2 NCO-Specific Adaptations

#### 4.2.1 Mapping to RL4CO Rollouts
| LLM Setting | NCO Translation |
|-------------|-----------------|
| Input x | CO instance (graph, cities, demands) |
| Sample y_i | Full solution trajectory τ_i |
| Reward g(y_i) | −tour_length or −cost |
| log p(y_i\|x) | Σ_t log π(a_t \| state_t) |

#### 4.2.2 Integration with POMO Multi-Start
- POMO generates N trajectories from different starting nodes
- Our estimator applies per-instance across these N trajectories
- Consider: should k = N, or k < N for different objectives?
- Note: Task 1 proofs assume i.i.d. trajectories; deterministic multi-start is
  exchangeable but not strictly i.i.d. For strict i.i.d. correspondence, sample
  start nodes randomly (with replacement) instead of enumerating them.

#### 4.2.3 Stability Mechanisms
- Gradient clipping (max norm)
- Advantage normalization
- Entropy regularization
- Reward scaling (handle reward sign: reward = −cost)
- Tie-breaking strategy for equal rewards (epsilon noise)

### 4.3 Algorithm Pseudocode

```python
def maxk_policy_gradient(policy, instances, n_samples, k):
    """
    Unbiased Max@K gradient estimator for NCO.
    
    Args:
        policy: Neural policy π_θ
        instances: Batch of CO instances
        n_samples: Number of samples per instance (n ≥ k)
        k: The K in Max@K objective
    """
    total_loss = 0
    
    for instance in instances:
        # 1. Sample n trajectories
        trajectories = [policy.sample(instance) for _ in range(n_samples)]
        rewards = torch.stack([compute_reward(τ) for τ in trajectories])  # [n_samples]
        log_probs = torch.stack([policy.log_prob(τ) for τ in trajectories])  # [n_samples]

        # 2. Compute per-sample gradient score-weights s_i for Max@k
        # IMPORTANT: this is NOT the same as the reward-estimator weights w_i.
        # See Task 1.2 for the closed form (includes the “Support” term).
        s_weights = maxk_gradient_weights(rewards, k)

        # 3. Apply variance reduction (choose one)
        # Sample-LOO: subtract b_i^LOO computed from rewards excluding i (requires n > k)
        # SubLOO: hitchhiking-free max–second-max gap form (requires k ≥ 2)
        s_weights = apply_sample_loo(s_weights, rewards, k)  # Sample-LOO
        # OR: s_weights = subloo_weights(rewards, k)          # SubLOO

        # 4. Compute loss (stop-gradient through weights)
        loss = -(s_weights.detach() * log_probs).sum()
        total_loss += loss
    
    return total_loss / len(instances)
```

---

## 5. Requirements

### 5.1 Functional Requirements

#### 5.1.1 Core Algorithm Components

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| FR1 | Implement unbiased Max@K reward estimator | P0 |
| FR2 | Implement unbiased Max@K gradient weights | P0 |
| FR3 | Implement LOO variance reduction baseline | P0 |
| FR4 | Support configurable k value (1 ≤ k ≤ n) | P0 |
| FR5 | Handle reward sorting with tie-breaking | P1 |
| FR6 | Integrate with RL4CO policy gradient framework | P1 |

#### 5.1.2 Training Infrastructure

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| FR7 | Support batch processing of multiple instances | P0 |
| FR8 | Implement gradient clipping | P1 |
| FR9 | Implement advantage normalization | P1 |
| FR10 | Support entropy regularization | P2 |
| FR11 | Logging of variance metrics | P1 |

#### 5.1.3 Evaluation

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| FR12 | Evaluate Max@K at same K as training | P0 |
| FR13 | Benchmark against Leader Reward baseline | P0 |
| FR14 | Benchmark against vanilla POMO | P1 |
| FR15 | Report optimality gaps | P1 |

### 5.2 Non-Functional Requirements

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| NFR1 | Computational overhead < 10% vs. Leader Reward | P1 |
| NFR2 | Compatible with PyTorch 2.0+ | P0 |
| NFR3 | Compatible with RL4CO library | P0 |
| NFR4 | Reproducible with fixed random seeds | P1 |
| NFR5 | GPU memory efficient (handle n=128 samples) | P1 |

---

## 6. Tasks & Milestones

### 6.1 Phase 1: Mathematical Foundation (Week 1-2)

| Task ID | Task | Deliverable |
|---------|------|-------------|
| T1.1 | Formal derivation of Max@K gradient estimator | Mathematical proof document |
| T1.2 | Prove unbiasedness property | Theorem + proof |
| T1.3 | Derive variance reduction via LOO | Mathematical formulation |
| T1.4 | Compare theoretical variance to Leader Reward | Analysis document |

### 6.2 Phase 2: Core Implementation (Week 3-4)

| Task ID | Task | Deliverable |
|---------|------|-------------|
| T2.1 | Implement MaxK reward estimator | `code/src/estimators/maxk_reward.py` |
| T2.2 | Implement gradient weight computation | `code/src/estimators/maxk_gradient.py` |
| T2.3 | Implement LOO baseline | `code/src/estimators/baselines.py` |
| T2.4 | Unit tests for all components | `code/tests/test_maxk_reward.py`, `code/tests/test_maxk_gradient.py`, `code/tests/test_baselines.py` |
| T2.5 | Validate against PKPO/RSPO formulas | `code/tests/test_pkpo_rspo_validation.py` |

### 6.3 Phase 3: RL4CO Integration (Week 5-6)

| Task ID | Task | Deliverable |
|---------|------|-------------|
| T3.1 | Create MaxK policy gradient class | RL4CO-compatible module |
| T3.2 | Integrate with POMO training loop | Training script |
| T3.3 | Implement stability mechanisms | Gradient clipping, normalization |
| T3.4 | Implement Leader Reward baseline | Comparison baseline |
| T3.5 | Create evaluation harness | Evaluation scripts |

### 6.4 Phase 4: Experiments & Validation (Week 7-10)

| Task ID | Task | Deliverable |
|---------|------|-------------|
| T4.1 | TSP experiments (50, 100 nodes) | Results tables, training curves |
| T4.2 | CVRP experiments | Results tables |
| T4.3 | OP experiments | Results tables |
| T4.4 | PCTSP experiments | Results tables |
| T4.5 | Variance analysis (empirical) | Variance plots |
| T4.6 | Convergence speed comparison | Training curves |
| T4.7 | Ablation: impact of k value | Ablation tables |
| T4.8 | Ablation: LOO baseline impact | Ablation results |

### 6.5 Phase 5: Paper Writing (Week 11-14)

| Task ID | Task | Deliverable |
|---------|------|-------------|
| T5.1 | Introduction draft | Paper section |
| T5.2 | Related work section | Paper section |
| T5.3 | Method section with proofs | Paper section |
| T5.4 | Experiments & results | Paper section |
| T5.5 | Conclusion & future work | Paper section |
| T5.6 | Figures and tables | Visualizations |
| T5.7 | Appendix (proofs, additional results) | Supplementary |

---

## 7. Success Metrics

### 7.1 Quantitative Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Variance reduction vs. Leader | > 30% lower variance | Empirical gradient variance |
| Convergence speedup | > 20% faster to same quality | Training epochs comparison |
| TSP-100 gap to optimal | 0.1-0.5% | Concorde comparison |
| CVRP gap to best known | < 5% | Literature comparison |
| Computational overhead | < 15% | Wall-clock time |

### 7.2 Qualitative Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| Mathematical rigor | Peer-review acceptance | Expert review |
| Code quality | Mergeable to RL4CO | Code review |
| Reproducibility | Full reproduction within 5% | Independent validation |

---

## 8. Risks & Mitigations

| Risk ID | Risk | Probability | Impact | Mitigation |
|---------|------|-------------|--------|------------|
| R1 | Variance reduction insufficient vs. Leader | Medium | High | Explore alternative baselines, entropy tuning |
| R2 | Computational overhead too high | Low | Medium | Optimize sorting, batch computation |
| R3 | Doesn't generalize to all RL4CO tasks | Medium | High | Start with TSP, incrementally add tasks |
| R4 | Training instability | Medium | Medium | Gradient clipping, careful hyperparameter sweep |
| R5 | Similar concurrent work published | Low | High | Move quickly, focus on CO-specific insights |

---

## 9. Implementation Details

### 9.1 Repository Structure

```
principled-bestofk/
├── code/
│   ├── requirements.txt          # Dependency list
│   ├── src/
│   │   ├── estimators/
│   │   │   ├── __init__.py
│   │   │   ├── maxk_reward.py        # Max@K reward estimator
│   │   │   ├── maxk_gradient.py      # Gradient weights
│   │   │   └── baselines.py          # LOO baselines
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   ├── maxk_reinforce.py     # MaxK REINFORCE
│   │   │   └── leader_reward.py      # Baseline implementation
│   │   ├── utils/
│   │   │   ├── sorting.py
│   │   │   └── combinatorics.py      # C(n,k) etc.
│   │   └── experiments/
│   │       ├── train_tsp.py
│   │       ├── train_cvrp.py
│   │       └── evaluate.py
│   └── tests/
│       ├── conftest.py
│       ├── test_maxk_reward.py
│       ├── test_maxk_gradient.py
│       ├── test_baselines.py
│       ├── test_pkpo_rspo_validation.py
│       └── test_stability.py
├── docs/
│   ├── PRD.md                    # This document
│   ├── description.txt
│   └── Tasks/
│       ├── Task1/
│       └── Task2/
├── knowledgebase/
│   ├── llm_context_maxk_rl4co.txt
│   └── papers/
├── notebooks/
│   ├── derivation_validation.ipynb
│   └── experiment_analysis.ipynb
└── configs/
    ├── tsp50.yaml
    ├── tsp100.yaml
    └── cvrp.yaml
```

### 9.2 Key Implementation Equations

#### 9.2.1 Two Different “Weights” (Reward vs. Gradient)

Task 1 established two related-but-distinct objects.

**(A) Reward-estimator weights (Task 1.1):**

These weights define an unbiased estimator of the Max@k *reward* using `n` samples.
For rewards sorted ascending, the unbiased reward estimator is
`ρ^(g)(n,k) = Σ_i w_i R_(i)` where:

```python
def compute_reward_weight(rank_i: int, n: int, k: int) -> float:
    """Reward-estimator weight w_i for the i-th order statistic.

    Args:
        rank_i: 1-indexed rank i after sorting rewards ascending.
        n: Number of samples.
        k: Max@k parameter.

    Returns:
        The scalar weight w_i.
    """
    if rank_i < k:
        return 0.0
    return comb(rank_i - 1, k - 1) / comb(n, k)
```

**(B) Gradient score-weights (Task 1.2):**

The unbiased Max@k *gradient* estimator has the form
`∇̂J_maxK = Σ_i s_i ∇ log π(τ_i)`.
For `k ≥ 2`, `s_i` is **not** equal to `w_i * R_(i)`; it includes a “Support” term
from subsets where a higher-ranked sample is the maximum.

Implementation should follow the closed form in
`docs/Tasks/Task1/task1.2/unbiasedness_proof.md` (Proposition 5.1).

#### 9.2.2 LOO Baselines (Variance Reduction)

Task 1 defines two variance-reduction options.

**Sample-LOO (requires `n > k`):**

Subtract a per-sample baseline `b_i^LOO` computed from the other `n-1` rewards:

```python
def sample_loo_weights(s_weights, rewards, n, k):
    """Sample-level leave-one-out variance reduction.

    Notes:
        This is the correct form: (s_i - b_i^LOO), where b_i^LOO does not depend
        on trajectory i. Do NOT use w_i * (R_i - b_i^LOO).

    Requires:
        n > k
    """
    b_loo = compute_maxk_reward_estimates_excluding_each_sample(rewards, n, k)
    return s_weights - b_loo
```

**SubLOO (requires `k ≥ 2`):**

Per-subset leave-one-out baseline yielding hitchhiking-free gap weights.
See `docs/Tasks/Task1/task1.3/loo_variance_reduction.md` (Proposition 2.1).

### 9.3 Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| k (Max@K) | 8 | [1, n] | Match to evaluation setting |
| n (samples) | 16 | [8, 128] | More samples = lower variance |
| Learning rate | 1e-4 | [1e-5, 1e-3] | Adam optimizer |
| Gradient clip | 1.0 | [0.5, 2.0] | Max norm clipping |
| Entropy coef | 0.01 | [0.001, 0.1] | Exploration |
| Batch size | 64 | [32, 256] | Instances per update |

---

## 10. Appendix

### 10.1 Key References

1. **REINFORCE**: Williams, "Simple statistical gradient-following algorithms" (1992)
2. **Neural CO**: Bello et al., "Neural Combinatorial Optimization with RL" (2017)
3. **Attention Model**: Kool et al., "Attention, Learn to Solve Routing Problems!" (ICLR 2019)
4. **POMO**: Kwon et al., "POMO: Policy Optimization with Multiple Optima" (NeurIPS 2020)
5. **Leader Reward**: Wang et al., "Leader Reward for POMO-Based NCO" (arXiv 2024)
6. **PKPO**: "Pass@K Policy Optimization: Solving Harder RL Problems" (arXiv 2025)
7. **RSPO**: Zhang et al., "Risk-Seeking Policy Optimization for Pass@k and Max@k" (arXiv 2025)
8. **RL4CO**: Berto et al., "RL4CO: Extensive RL for CO Benchmark" (arXiv 2023)

### 10.2 Glossary

| Term | Definition |
|------|------------|
| Max@K | Objective: expected maximum reward among K i.i.d. samples |
| Pass@K | Probability at least one success among K samples (binary) |
| Hitchhiking | Bias from reinforcing bad samples that co-occurred with good ones |
| LOO | Leave-one-out (variance reduction technique) |
| NCO | Neural Combinatorial Optimization |
| POMO | Policy Optimization with Multiple Optima |
| RL4CO | Reinforcement Learning for Combinatorial Optimization (library) |

### 10.3 Related Conversation History
- **Create RL4CO Knowledge Base** (Jan 9, 2026): Created comprehensive knowledge base from NeurIPS papers
- **Update Claude Skill for RL4CO** (Jan 9, 2026): Updated agent skills for project planning

---

**Document Owner:** Research Team  
**Next Review:** Week 2 checkpoint  
**Approval Status:** Pending
