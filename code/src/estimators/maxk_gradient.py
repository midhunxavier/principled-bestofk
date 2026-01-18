"""Unbiased Max@K gradient score-weight computation.

Implements Proposition 5.1 from Task 1.2 (unbiasedness_proof.md):

    s_{(i)} = (1 / C(n, K)) * [
        1[i >= K] * C(i-1, K-1) * R_{(i)}
        + sum_{j=i+1..n} C(j-2, K-2) * R_{(j)}
    ]

where R_{(i)} are rewards sorted ascending (order statistics).

References:
    - docs/Tasks/Task1/task1.2/unbiasedness_proof.md (Proposition 5.1)
    - docs/PRD.md §9.2.1
"""

from __future__ import annotations

import math
from typing import Optional

import torch

_COEFF_VALUES_CACHE: dict[tuple[int, int], tuple[tuple[float, ...], tuple[float, ...]]] = {}
_COEFF_TENSOR_CACHE: dict[
    tuple[int, int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


def _device_to_key(device: Optional[torch.device]) -> str:
    return "cpu" if device is None else str(device)


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    """Raise ValueError if dtype is not floating point."""
    try:
        torch.finfo(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dtype must be floating point, got {dtype}") from exc


def _gradient_coefficients(
    n: int,
    k: int,
    *,
    device: Optional[torch.device],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached win/support coefficient tensors for given (n, k)."""
    values_key = (n, k)
    coeff_values = _COEFF_VALUES_CACHE.get(values_key)
    if coeff_values is None:
        win_coeff = [0.0] * n
        support_coeff = [0.0] * n
        for idx in range(n):
            rank = idx + 1
            if rank >= k:
                win_coeff[idx] = math.comb(rank - 1, k - 1)
            if rank >= 2:
                support_coeff[idx] = math.comb(rank - 2, k - 2)
        coeff_values = (tuple(win_coeff), tuple(support_coeff))
        _COEFF_VALUES_CACHE[values_key] = coeff_values

    tensor_key = (n, k, _device_to_key(device), dtype)
    coeff_tensors = _COEFF_TENSOR_CACHE.get(tensor_key)
    if coeff_tensors is None:
        win_coeff = torch.tensor(coeff_values[0], dtype=dtype)
        support_coeff = torch.tensor(coeff_values[1], dtype=dtype)
        if device is not None:
            win_coeff = win_coeff.to(device)
            support_coeff = support_coeff.to(device)
        coeff_tensors = (win_coeff, support_coeff)
        _COEFF_TENSOR_CACHE[tensor_key] = coeff_tensors

    return coeff_tensors


def maxk_gradient_weights(
    rewards: torch.Tensor,
    k: int,
    *,
    stable_sort: bool = True,
) -> torch.Tensor:
    """Compute unbiased Max@K gradient score weights for each sample.

    The estimator has the form:

        G_hat = sum_i s_i * grad log pi(tau_i),

    where s_i depends on all rewards. For k >= 2, the weights include support
    terms from subsets where higher-ranked samples are the maximum.

    Args:
        rewards: Tensor of shape [n] or [batch, n] containing reward values.
            Non-floating tensors are converted to float64 for computation.
        k: The K in Max@K objective (1 <= k <= n).
        stable_sort: If True, use stable sorting (deterministic tie-breaking by
            original index). Default True.

    Returns:
        Tensor of same shape as rewards containing score weights s_i aligned
        with the original (unsorted) sample order. Output dtype matches input
        for floating tensors; otherwise it is float64.

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

    if k == 1:
        weights = compute_rewards / n
        return weights.squeeze(0) if squeeze_output else weights

    dtype = compute_rewards.dtype
    _validate_floating_dtype(dtype)

    sorted_rewards, sorted_indices = torch.sort(
        compute_rewards, dim=-1, stable=stable_sort
    )

    win_coeff, support_coeff = _gradient_coefficients(
        n, k, device=sorted_rewards.device, dtype=dtype
    )

    win_term = win_coeff * sorted_rewards
    weighted_support = support_coeff * sorted_rewards
    suffix_inclusive = torch.cumsum(weighted_support.flip(-1), dim=-1).flip(-1)

    support_term = torch.zeros_like(sorted_rewards)
    support_term[..., :-1] = suffix_inclusive[..., 1:]

    normalizer = math.comb(n, k)
    s_sorted = (win_term + support_term) / normalizer

    weights = torch.zeros_like(s_sorted)
    weights.scatter_(-1, sorted_indices, s_sorted)

    return weights.squeeze(0) if squeeze_output else weights
