"""Leave-one-out baselines for Max@K gradient estimators.

Implements Sample-LOO and SubLOO baselines from Task 1.3:

Sample-LOO baseline (Definition 2.1):
    b_i^LOO = (1 / C(n-1, K)) * sum_{S subset [n]\\{i}, |S|=K} max_{j in S} R_j

SubLOO weights (Proposition 2.1):
    \tilde{s}_{(i)} = (1 / C(n, K)) * sum_{m=K..i} C(m-2, K-2)
                     * (R_{(i)} - R_{(m-1)}) * 1[i >= K]

References:
    - docs/Tasks/Task1/task1.3/loo_variance_reduction.md
    - docs/PRD.md §9.2.2
"""

from __future__ import annotations

import math

import torch


def sample_loo_baseline(
    rewards: torch.Tensor,
    k: int,
    *,
    stable_sort: bool = True,
) -> torch.Tensor:
    """Compute Sample-LOO baselines for each sample.

    Args:
        rewards: Tensor of shape [n] or [batch, n] containing reward values.
            Non-floating tensors are converted to float64 for computation.
        k: The K in Max@K objective (1 <= k < n).
        stable_sort: If True, use stable sorting (deterministic tie-breaking).

    Returns:
        Tensor of shape [n] or [batch, n] with b_i^LOO values aligned to the
        original sample order.

    Raises:
        ValueError: If rewards has invalid shape or k is out of range.
    """
    squeeze_output = False
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
        squeeze_output = True

    if rewards.ndim != 2:
        raise ValueError(f"rewards must be 1D or 2D, got ndim={rewards.ndim}")

    n = rewards.shape[-1]
    if k < 1 or k >= n:
        raise ValueError(f"k must satisfy 1 <= k < n, got k={k}, n={n}")

    compute_rewards = rewards
    if not compute_rewards.is_floating_point():
        compute_rewards = compute_rewards.to(torch.float64)

    sorted_rewards, sorted_indices = torch.sort(
        compute_rewards, dim=-1, stable=stable_sort
    )

    dtype = sorted_rewards.dtype
    device = sorted_rewards.device

    coeff_keep = torch.zeros(n, dtype=dtype, device=device)
    coeff_shift = torch.zeros(n, dtype=dtype, device=device)
    for rank in range(1, n + 1):
        if rank >= k:
            coeff_keep[rank - 1] = math.comb(rank - 1, k - 1)
        if rank >= k + 1:
            coeff_shift[rank - 1] = math.comb(rank - 2, k - 1)

    keep_weighted = sorted_rewards * coeff_keep
    shift_weighted = sorted_rewards * coeff_shift

    prefix_keep = torch.zeros(
        (*sorted_rewards.shape[:-1], n + 1), dtype=dtype, device=device
    )
    prefix_shift = torch.zeros_like(prefix_keep)

    prefix_keep[..., 1:] = torch.cumsum(keep_weighted, dim=-1)
    prefix_shift[..., 1:] = torch.cumsum(shift_weighted, dim=-1)

    norm = math.comb(n - 1, k)
    base_common = (prefix_shift[..., n] - prefix_shift[..., k - 1]) / norm

    ranks = torch.arange(n, device=device)
    base_keep = prefix_keep[..., k - 1].unsqueeze(-1)
    sum_keep = prefix_keep[..., ranks] - base_keep
    sum_shift = prefix_shift[..., n].unsqueeze(-1) - prefix_shift[..., ranks + 1]
    baseline_sorted = (sum_keep + sum_shift) / norm

    baseline_sorted = torch.where(
        ranks < k - 1, base_common.unsqueeze(-1), baseline_sorted
    )

    baselines = torch.zeros_like(baseline_sorted)
    baselines.scatter_(-1, sorted_indices, baseline_sorted)

    return baselines.squeeze(0) if squeeze_output else baselines


def apply_sample_loo(
    s_weights: torch.Tensor,
    rewards: torch.Tensor,
    k: int,
    *,
    stable_sort: bool = True,
) -> torch.Tensor:
    """Subtract Sample-LOO baselines from gradient weights.

    Args:
        s_weights: Tensor of shape [n] or [batch, n] with gradient score weights.
        rewards: Tensor of shape [n] or [batch, n] containing reward values.
        k: The K in Max@K objective (1 <= k < n).
        stable_sort: If True, use stable sorting (deterministic tie-breaking).

    Returns:
        Tensor of same shape as s_weights containing baseline-subtracted weights.

    Raises:
        ValueError: If rewards and s_weights shapes mismatch.
    """
    if rewards.shape != s_weights.shape:
        raise ValueError(
            "rewards and s_weights must have the same shape, "
            f"got rewards={rewards.shape}, s_weights={s_weights.shape}"
        )

    baselines = sample_loo_baseline(rewards, k, stable_sort=stable_sort)
    return s_weights - baselines


def subloo_weights(
    rewards: torch.Tensor,
    k: int,
    *,
    stable_sort: bool = True,
) -> torch.Tensor:
    """Compute SubLOO weights that eliminate hitchhiking.

    Args:
        rewards: Tensor of shape [n] or [batch, n] containing reward values.
            Non-floating tensors are converted to float64 for computation.
        k: The K in Max@K objective (2 <= k <= n).
        stable_sort: If True, use stable sorting (deterministic tie-breaking).

    Returns:
        Tensor of shape [n] or [batch, n] with SubLOO weights aligned to the
        original sample order.

    Raises:
        ValueError: If rewards has invalid shape or k is out of range.
    """
    squeeze_output = False
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
        squeeze_output = True

    if rewards.ndim != 2:
        raise ValueError(f"rewards must be 1D or 2D, got ndim={rewards.ndim}")

    n = rewards.shape[-1]
    if k < 2 or k > n:
        raise ValueError(f"k must satisfy 2 <= k <= n, got k={k}, n={n}")

    compute_rewards = rewards
    if not compute_rewards.is_floating_point():
        compute_rewards = compute_rewards.to(torch.float64)

    sorted_rewards, sorted_indices = torch.sort(
        compute_rewards, dim=-1, stable=stable_sort
    )

    dtype = sorted_rewards.dtype
    device = sorted_rewards.device

    coeff_rank = torch.zeros(n + 1, dtype=dtype, device=device)
    for rank in range(k, n + 1):
        coeff_rank[rank] = math.comb(rank - 2, k - 2)

    prefix_coeff = torch.cumsum(coeff_rank, dim=-1)

    prefix_weighted = torch.zeros(
        (*sorted_rewards.shape[:-1], n + 1), dtype=dtype, device=device
    )
    weighted = torch.zeros_like(sorted_rewards)
    if n > 1:
        weighted[..., 1:] = coeff_rank[2:] * sorted_rewards[..., :-1]
    prefix_weighted[..., 1:] = torch.cumsum(weighted, dim=-1)

    norm = math.comb(n, k)
    base_prefix = prefix_coeff[k - 1]
    base_weighted = prefix_weighted[..., k - 1].unsqueeze(-1)

    ranks = torch.arange(1, n + 1, device=device)
    num_subsets = prefix_coeff[ranks] - base_prefix
    weighted_sum = prefix_weighted[..., ranks] - base_weighted

    sum_gaps = sorted_rewards * num_subsets - weighted_sum
    weights_sorted = sum_gaps / norm
    weights_sorted = torch.where(
        ranks < k, torch.zeros_like(weights_sorted), weights_sorted
    )

    weights = torch.zeros_like(weights_sorted)
    weights.scatter_(-1, sorted_indices, weights_sorted)

    return weights.squeeze(0) if squeeze_output else weights
