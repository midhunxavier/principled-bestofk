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

_INTERNAL_DTYPE = torch.float64

_COEFF_VALUES_CACHE: dict[tuple[int, int], tuple[tuple[float, ...], tuple[float, ...]]] = {}
_COEFF_TENSOR_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _device_to_key(device: Optional[torch.device]) -> str:
    return "cpu" if device is None else str(device)


def _gradient_coefficients(
    n: int,
    k: int,
    *,
    device: Optional[torch.device],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached win/support coefficient tensors for given (n, k).

    Coefficients are always stored as float64 to avoid overflow in float32 when
    n is large (e.g., n=128) and rewards are not tiny.

    Args:
        n: Number of samples.
        k: The K in Max@K objective.
        device: Target device for the tensors.

    Returns:
        Tuple (win_coeff, support_coeff), each a float64 tensor of shape [n].
    """
    values_key = (n, k)
    coeff_values = _COEFF_VALUES_CACHE.get(values_key)
    if coeff_values is None:
        win_coeff_values: list[float] = [0.0] * n
        support_coeff_values: list[float] = [0.0] * n
        for idx in range(n):
            rank = idx + 1
            if rank >= k:
                win_coeff_values[idx] = math.comb(rank - 1, k - 1)
            if rank >= 2:
                support_coeff_values[idx] = math.comb(rank - 2, k - 2)
        coeff_values = (tuple(win_coeff_values), tuple(support_coeff_values))
        _COEFF_VALUES_CACHE[values_key] = coeff_values

    tensor_key = (n, k, _device_to_key(device))
    coeff_tensors = _COEFF_TENSOR_CACHE.get(tensor_key)
    if coeff_tensors is None:
        kwargs = {"dtype": _INTERNAL_DTYPE}
        if device is not None:
            kwargs["device"] = device
        win_coeff_tensor = torch.tensor(coeff_values[0], **kwargs)
        support_coeff_tensor = torch.tensor(coeff_values[1], **kwargs)
        coeff_tensors = (win_coeff_tensor, support_coeff_tensor)
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
            For numerical stability, computations for k>=2 are performed in
            float64 regardless of input dtype, and cast back at the end.
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

    output_dtype = rewards.dtype if rewards.is_floating_point() else torch.float64

    compute_rewards = rewards
    if not compute_rewards.is_floating_point():
        compute_rewards = compute_rewards.to(_INTERNAL_DTYPE)

    if k == 1:
        weights = compute_rewards / n
        if weights.dtype != output_dtype:
            weights = weights.to(output_dtype)
        return weights.squeeze(0) if squeeze_output else weights

    sorted_rewards, sorted_indices = torch.sort(
        compute_rewards, dim=-1, stable=stable_sort
    )
    sorted_rewards = sorted_rewards.to(_INTERNAL_DTYPE)

    win_coeff, support_coeff = _gradient_coefficients(
        n, k, device=sorted_rewards.device
    )

    win_term = win_coeff * sorted_rewards
    weighted_support = support_coeff * sorted_rewards
    suffix_inclusive = torch.cumsum(weighted_support.flip(-1), dim=-1).flip(-1)

    support_term = torch.zeros_like(sorted_rewards)
    support_term[..., :-1] = suffix_inclusive[..., 1:]

    inv_normalizer = 1.0 / math.comb(n, k)
    s_sorted = (win_term + support_term) * inv_normalizer

    weights = torch.zeros_like(s_sorted)
    weights.scatter_(-1, sorted_indices, s_sorted)

    if weights.dtype != output_dtype:
        weights = weights.to(output_dtype)

    return weights.squeeze(0) if squeeze_output else weights
