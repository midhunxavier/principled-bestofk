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
from typing import Any, Optional

import torch

_INTERNAL_DTYPE = torch.float64

_SAMPLE_LOO_COEFF_VALUES_CACHE: dict[
    tuple[int, int], tuple[tuple[float, ...], tuple[float, ...]]
] = {}
_SAMPLE_LOO_COEFF_TENSOR_CACHE: dict[
    tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]
] = {}

_SUBLOO_COEFF_VALUES_CACHE: dict[tuple[int, int], tuple[float, ...]] = {}
_SUBLOO_COEFF_TENSOR_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _device_to_key(device: Optional[torch.device]) -> str:
    return "cpu" if device is None else str(device)


def _sample_loo_coefficients(
    n: int,
    k: int,
    *,
    device: Optional[torch.device],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached Sample-LOO coefficient tensors for given (n, k).

    Coefficients are stored as float64 to avoid overflow in float32 when n is
    large (e.g., n=128) and rewards are not tiny.

    Args:
        n: Number of samples.
        k: The K in Max@K objective (1 <= k < n).
        device: Target device for the tensors.

    Returns:
        Tuple (coeff_keep, coeff_shift), each a float64 tensor of shape [n].
    """
    values_key = (n, k)
    coeff_values = _SAMPLE_LOO_COEFF_VALUES_CACHE.get(values_key)
    if coeff_values is None:
        coeff_keep_values: list[float] = [0.0] * n
        coeff_shift_values: list[float] = [0.0] * n
        for rank in range(1, n + 1):
            if rank >= k:
                coeff_keep_values[rank - 1] = math.comb(rank - 1, k - 1)
            if rank >= k + 1:
                coeff_shift_values[rank - 1] = math.comb(rank - 2, k - 1)
        coeff_values = (tuple(coeff_keep_values), tuple(coeff_shift_values))
        _SAMPLE_LOO_COEFF_VALUES_CACHE[values_key] = coeff_values

    tensor_key = (n, k, _device_to_key(device))
    coeff_tensors = _SAMPLE_LOO_COEFF_TENSOR_CACHE.get(tensor_key)
    if coeff_tensors is None:
        kwargs: dict[str, Any] = {"dtype": _INTERNAL_DTYPE}
        if device is not None:
            kwargs["device"] = device
        coeff_keep = torch.tensor(coeff_values[0], **kwargs)
        coeff_shift = torch.tensor(coeff_values[1], **kwargs)
        coeff_tensors = (coeff_keep, coeff_shift)
        _SAMPLE_LOO_COEFF_TENSOR_CACHE[tensor_key] = coeff_tensors

    return coeff_tensors


def _subloo_coefficients(
    n: int,
    k: int,
    *,
    device: Optional[torch.device],
) -> torch.Tensor:
    """Return cached SubLOO coefficient tensor coeff_rank for given (n, k).

    Args:
        n: Number of samples.
        k: The K in Max@K objective (2 <= k <= n).
        device: Target device for the tensor.

    Returns:
        Float64 tensor coeff_rank of shape [n + 1] with coeff_rank[rank] =
        C(rank-2, k-2) for rank in [k..n], and 0 elsewhere.
    """
    values_key = (n, k)
    coeff_values = _SUBLOO_COEFF_VALUES_CACHE.get(values_key)
    if coeff_values is None:
        values: list[float] = [0.0] * (n + 1)
        for rank in range(k, n + 1):
            values[rank] = math.comb(rank - 2, k - 2)
        coeff_values = tuple(values)
        _SUBLOO_COEFF_VALUES_CACHE[values_key] = coeff_values

    tensor_key = (n, k, _device_to_key(device))
    coeff_rank = _SUBLOO_COEFF_TENSOR_CACHE.get(tensor_key)
    if coeff_rank is None:
        kwargs: dict[str, Any] = {"dtype": _INTERNAL_DTYPE}
        if device is not None:
            kwargs["device"] = device
        coeff_rank = torch.tensor(coeff_values, **kwargs)
        _SUBLOO_COEFF_TENSOR_CACHE[tensor_key] = coeff_rank

    return coeff_rank


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
            For numerical stability, computations are performed in float64 and
            cast back to the input dtype at the end.
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

    output_dtype = rewards.dtype if rewards.is_floating_point() else _INTERNAL_DTYPE

    compute_rewards = rewards
    if not compute_rewards.is_floating_point():
        compute_rewards = compute_rewards.to(_INTERNAL_DTYPE)

    sorted_rewards, sorted_indices = torch.sort(
        compute_rewards, dim=-1, stable=stable_sort
    )
    sorted_rewards = sorted_rewards.to(_INTERNAL_DTYPE)

    device = sorted_rewards.device

    coeff_keep, coeff_shift = _sample_loo_coefficients(n, k, device=device)

    keep_weighted = sorted_rewards * coeff_keep
    shift_weighted = sorted_rewards * coeff_shift

    prefix_keep = torch.zeros(
        (*sorted_rewards.shape[:-1], n + 1), dtype=_INTERNAL_DTYPE, device=device
    )
    prefix_shift = torch.zeros_like(prefix_keep)

    prefix_keep[..., 1:] = torch.cumsum(keep_weighted, dim=-1)
    prefix_shift[..., 1:] = torch.cumsum(shift_weighted, dim=-1)

    inv_norm = 1.0 / math.comb(n - 1, k)
    base_common = (prefix_shift[..., n] - prefix_shift[..., k - 1]) * inv_norm

    ranks = torch.arange(n, device=device)
    base_keep = prefix_keep[..., k - 1].unsqueeze(-1)
    sum_keep = prefix_keep[..., ranks] - base_keep
    sum_shift = prefix_shift[..., n].unsqueeze(-1) - prefix_shift[..., ranks + 1]
    baseline_sorted = (sum_keep + sum_shift) * inv_norm

    baseline_sorted = torch.where(
        ranks < k - 1, base_common.unsqueeze(-1), baseline_sorted
    )

    baselines = torch.zeros_like(baseline_sorted)
    baselines.scatter_(-1, sorted_indices, baseline_sorted)

    if baselines.dtype != output_dtype:
        baselines = baselines.to(output_dtype)

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
    min_gap_scale: float = 0.0,
) -> torch.Tensor:
    """Compute SubLOO weights that eliminate hitchhiking.

    Args:
        rewards: Tensor of shape [n] or [batch, n] containing reward values.
            Non-floating tensors are converted to float64 for computation.
            For numerical stability, computations are performed in float64 and
            cast back to the input dtype at the end.
        k: The K in Max@K objective (2 <= k <= n).
        stable_sort: If True, use stable sorting (deterministic tie-breaking).
        min_gap_scale: Minimum gap as a fraction of the reward range. This prevents
            zero gradients when rewards are clustered (e.g., early in training).
            The minimum gap is computed as: min_gap = min_gap_scale * (R_max - R_min).
            Set to 0.0 to disable (original behavior). Recommended: 0.01-0.05.

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

    output_dtype = rewards.dtype if rewards.is_floating_point() else _INTERNAL_DTYPE

    compute_rewards = rewards
    if not compute_rewards.is_floating_point():
        compute_rewards = compute_rewards.to(_INTERNAL_DTYPE)

    sorted_rewards, sorted_indices = torch.sort(
        compute_rewards, dim=-1, stable=stable_sort
    )
    sorted_rewards = sorted_rewards.to(_INTERNAL_DTYPE)

    device = sorted_rewards.device

    coeff_rank = _subloo_coefficients(n, k, device=device)
    prefix_coeff = torch.cumsum(coeff_rank, dim=-1)

    prefix_weighted = torch.zeros(
        (*sorted_rewards.shape[:-1], n + 1), dtype=_INTERNAL_DTYPE, device=device
    )
    weighted = torch.zeros_like(sorted_rewards)
    if n > 1:
        weighted[..., 1:] = coeff_rank[2:] * sorted_rewards[..., :-1]
    prefix_weighted[..., 1:] = torch.cumsum(weighted, dim=-1)

    inv_norm = 1.0 / math.comb(n, k)
    base_prefix = prefix_coeff[k - 1]
    base_weighted = prefix_weighted[..., k - 1].unsqueeze(-1)

    ranks = torch.arange(1, n + 1, device=device)
    num_subsets = prefix_coeff[ranks] - base_prefix
    weighted_sum = prefix_weighted[..., ranks] - base_weighted

    sum_gaps = sorted_rewards * num_subsets - weighted_sum
    
    # Apply minimum gap floor to prevent zero gradients when rewards are clustered
    # This ensures gradient signal even in early training when policy is random
    if min_gap_scale > 0:
        reward_range = sorted_rewards[..., -1:] - sorted_rewards[..., :1]  # [batch, 1]
        min_gap = min_gap_scale * reward_range  # [batch, 1]
        # Only apply to ranks >= k (those that get gradient)
        is_top_k = (ranks >= k).float()  # [n]
        # Ensure each contributing sample has at least min_gap contribution
        # Scale by num_subsets to maintain proper relative weighting
        min_contribution = min_gap * num_subsets * is_top_k
        sum_gaps = torch.maximum(sum_gaps, min_contribution)
    
    weights_sorted = sum_gaps * inv_norm
    weights_sorted = torch.where(
        ranks < k, torch.zeros_like(weights_sorted), weights_sorted
    )

    weights = torch.zeros_like(weights_sorted)
    weights.scatter_(-1, sorted_indices, weights_sorted)

    if weights.dtype != output_dtype:
        weights = weights.to(output_dtype)

    return weights.squeeze(0) if squeeze_output else weights
