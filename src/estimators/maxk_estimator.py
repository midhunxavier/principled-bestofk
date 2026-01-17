"""Unbiased Max@K reward estimator.

Implements Theorem 3.1 from Task 1.1 (mathematical_derivation.md):

    ρ̂^(g)(n, K) = (1 / C(n,K)) * Σ_{i=K}^{n} C(i-1, K-1) * R_(i)

where R_(i) are order statistics (sorted ascending) and C(a,b) = binomial(a,b).

References:
    - docs/Tasks/Task1/task1.1/mathematical_derivation.md (Theorem 3.1)
    - docs/PRD.md §9.2.1
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def maxk_reward_weights(
    n: int,
    k: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute reward-estimator weights for each order-statistic rank.

    For rewards sorted ascending into R_(1) <= ... <= R_(n), the weight w_i
    for the i-th order statistic (1-indexed) is:

        w_i = C(i-1, k-1) / C(n, k)   if i >= k
        w_i = 0                        if i < k

    The returned tensor uses 0-based indexing: weights[j] corresponds to
    rank i = j + 1 in the derivation.

    Args:
        n: Number of samples.
        k: The K in Max@K (1 <= k <= n).
        device: Target device for the tensor.
        dtype: Target dtype (defaults to float64 for numerical precision).

    Returns:
        Tensor of shape [n] with weights summing to 1.

    Raises:
        ValueError: If k < 1 or k > n.
    """
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")

    if dtype is None:
        dtype = torch.float64

    # Compute C(n, k) once
    c_n_k = math.comb(n, k)

    weights = torch.zeros(n, dtype=dtype, device=device)

    # Fill weights for ranks k..n (1-indexed), i.e., indices k-1..n-1 (0-indexed)
    for i in range(k, n + 1):  # i is 1-indexed rank
        c_i_minus_1_k_minus_1 = math.comb(i - 1, k - 1)
        weights[i - 1] = c_i_minus_1_k_minus_1 / c_n_k

    return weights


def maxk_reward_estimate(
    rewards: torch.Tensor,
    k: int,
    *,
    stable_sort: bool = True,
) -> torch.Tensor:
    """Compute unbiased Max@K reward estimate.

    Given n samples with rewards, returns an unbiased estimate of
    E[max_{i in 1..k} R_i] where the expectation is over fresh k i.i.d. samples.

    Implements Theorem 3.1 (Task 1.1):

        ρ̂^(g)(n, K) = Σ_{i=1}^{n} w_i * R_(i)

    where R_(i) are sorted ascending and w_i are the binomial weights.

    Args:
        rewards: Tensor of shape [n] or [batch, n] containing reward values.
        k: The K in Max@K objective (1 <= k <= n).
        stable_sort: If True, use stable sorting (deterministic tie-breaking by
            original index). Default True.

    Returns:
        Scalar tensor if input is [n], or tensor of shape [batch] if input is
        [batch, n].

    Raises:
        ValueError: If k is out of valid range.
    """
    # Handle 1D input by adding batch dimension
    squeeze_output = False
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
        squeeze_output = True

    if rewards.ndim != 2:
        raise ValueError(f"rewards must be 1D or 2D, got ndim={rewards.ndim}")

    batch_size, n = rewards.shape

    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")

    # Sort rewards ascending along sample dimension
    sorted_rewards, _ = torch.sort(rewards, dim=-1, stable=stable_sort)

    # Get weights (computed on CPU with math.comb, then moved to device)
    weights = maxk_reward_weights(n, k, device=rewards.device, dtype=rewards.dtype)

    # Compute weighted sum: [batch, n] * [n] -> [batch, n] -> sum -> [batch]
    estimate = (sorted_rewards * weights).sum(dim=-1)

    if squeeze_output:
        estimate = estimate.squeeze(0)

    return estimate
