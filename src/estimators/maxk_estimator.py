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

# Module-level caches to avoid recomputing binomial weights.
#
# Keys:
# - _WEIGHT_VALUES_CACHE: (n, k) -> tuple of Python floats (CPU)
# - _WEIGHT_TENSOR_CACHE: (n, k, device_str, dtype) -> torch.Tensor
#
# Note: Returned tensors should be treated as read-only. If callers mutate the
# returned tensor in-place, future calls may observe the mutation.
_WEIGHT_VALUES_CACHE: dict[tuple[int, int], tuple[float, ...]] = {}
_WEIGHT_TENSOR_CACHE: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}


def _device_to_key(device: Optional[torch.device]) -> str:
    return "cpu" if device is None else str(device)


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    """Raise ValueError if dtype is not floating point."""
    try:
        torch.finfo(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dtype must be floating point, got {dtype}") from exc


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
        dtype: Floating dtype for the returned tensor (defaults to float64).

    Returns:
        Tensor of shape [n] with weights summing to 1.

    Raises:
        ValueError: If n < 1, if k is out of range, or if dtype is not floating.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got n={n}")
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")

    if dtype is None:
        dtype = torch.float64
    _validate_floating_dtype(dtype)

    values_key = (n, k)
    weights_values = _WEIGHT_VALUES_CACHE.get(values_key)
    if weights_values is None:
        c_n_k = math.comb(n, k)

        # Build weights on CPU as Python floats.
        weights_list = [0.0] * n
        for i in range(k, n + 1):
            weights_list[i - 1] = math.comb(i - 1, k - 1) / c_n_k
        weights_values = tuple(weights_list)
        _WEIGHT_VALUES_CACHE[values_key] = weights_values

    tensor_key = (n, k, _device_to_key(device), dtype)
    weights = _WEIGHT_TENSOR_CACHE.get(tensor_key)
    if weights is None:
        weights = torch.tensor(weights_values, dtype=dtype)
        if device is not None:
            weights = weights.to(device)
        _WEIGHT_TENSOR_CACHE[tensor_key] = weights

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
            Non-floating tensors are converted to float64 for computation.
        k: The K in Max@K objective (1 <= k <= n).
        stable_sort: If True, use stable sorting (deterministic tie-breaking by
            original index). Default True.

    Returns:
        Scalar tensor if input is [n], or tensor of shape [batch] if input is
        [batch, n]. Output dtype matches input for floating tensors; otherwise
        it is float64.

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

    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")

    compute_rewards = rewards
    if not compute_rewards.is_floating_point():
        compute_rewards = compute_rewards.to(torch.float64)

    sorted_rewards, _ = torch.sort(compute_rewards, dim=-1, stable=stable_sort)

    weights = maxk_reward_weights(
        n, k, device=sorted_rewards.device, dtype=sorted_rewards.dtype
    )

    estimate = (sorted_rewards * weights).sum(dim=-1)

    if squeeze_output:
        estimate = estimate.squeeze(0)

    return estimate
