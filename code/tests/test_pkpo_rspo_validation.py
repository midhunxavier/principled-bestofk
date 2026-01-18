"""Numerical validation against PKPO/RSPO closed-form formulas.

Task T2.5 asks for a sanity check that our implementations match the
closed-form combinatorial expressions derived in PKPO/RSPO-style work.

In this repo, the canonical derivations live in:
  - docs/Tasks/Task1/task1.1/mathematical_derivation.md (Max@K reward estimator)
  - docs/Tasks/Task1/task1.2/unbiasedness_proof.md (Prop 5.1: gradient weights)
  - docs/Tasks/Task1/task1.3/loo_variance_reduction.md (Sample-LOO + SubLOO)
"""

from __future__ import annotations

import math

import pytest
import torch

from src.estimators.baselines import sample_loo_baseline, subloo_weights
from src.estimators.maxk_estimator import maxk_reward_estimate
from src.estimators.maxk_gradient import maxk_gradient_weights


def _as_2d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 1:
        return x.unsqueeze(0), True
    return x, False


def pkpo_maxgk_reward_estimate_reference(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Reference implementation of PKPO maxg@k reward estimator (closed form).

    Args:
        rewards: Tensor of shape [n] or [batch, n].
        k: The K in Max@K (1 <= k <= n).

    Returns:
        Scalar tensor if input is [n], or [batch] if input is [batch, n].
    """
    rewards_2d, squeeze = _as_2d(rewards)
    if rewards_2d.ndim != 2:
        raise ValueError("rewards must be 1D or 2D")

    n = rewards_2d.shape[-1]
    if k < 1 or k > n:
        raise ValueError("k out of range")

    sorted_rewards, _ = torch.sort(rewards_2d, dim=-1, stable=True)
    norm = math.comb(n, k)

    weights = torch.zeros(n, dtype=sorted_rewards.dtype, device=sorted_rewards.device)
    for rank in range(k, n + 1):
        weights[rank - 1] = math.comb(rank - 1, k - 1) / norm

    estimate = (sorted_rewards * weights).sum(dim=-1)
    return estimate.squeeze(0) if squeeze else estimate


def rspo_maxk_gradient_weights_reference(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Reference implementation of RSPO Max@k gradient score weights (Prop 5.1).

    Args:
        rewards: Tensor of shape [n] or [batch, n].
        k: The K in Max@K (1 <= k <= n).

    Returns:
        Tensor of shape [n] or [batch, n] with weights aligned to original order.
    """
    rewards_2d, squeeze = _as_2d(rewards)
    if rewards_2d.ndim != 2:
        raise ValueError("rewards must be 1D or 2D")

    n = rewards_2d.shape[-1]
    if k < 1 or k > n:
        raise ValueError("k out of range")

    if k == 1:
        weights = rewards_2d / n
        return weights.squeeze(0) if squeeze else weights

    sorted_rewards, sorted_indices = torch.sort(rewards_2d, dim=-1, stable=True)
    norm = math.comb(n, k)

    s_sorted = torch.zeros_like(sorted_rewards)
    for i_idx in range(n):
        rank_i = i_idx + 1
        win_term = 0.0
        if rank_i >= k:
            win_term = math.comb(rank_i - 1, k - 1) * sorted_rewards[..., i_idx]

        support_term = 0.0
        for j_idx in range(i_idx + 1, n):
            rank_j = j_idx + 1
            support_term = support_term + math.comb(rank_j - 2, k - 2) * sorted_rewards[
                ..., j_idx
            ]

        s_sorted[..., i_idx] = (win_term + support_term) / norm

    weights = torch.zeros_like(s_sorted)
    weights.scatter_(-1, sorted_indices, s_sorted)
    return weights.squeeze(0) if squeeze else weights


def pkpo_sample_loo_baseline_reference(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Reference implementation of Sample-LOO baseline closed form (Task 1.3 §2.1.1).

    Args:
        rewards: Tensor of shape [n] or [batch, n].
        k: The K in Max@K (1 <= k < n).

    Returns:
        Tensor of shape [n] or [batch, n] aligned to original order.
    """
    rewards_2d, squeeze = _as_2d(rewards)
    if rewards_2d.ndim != 2:
        raise ValueError("rewards must be 1D or 2D")

    n = rewards_2d.shape[-1]
    if k < 1 or k >= n:
        raise ValueError("k out of range")

    sorted_rewards, sorted_indices = torch.sort(rewards_2d, dim=-1, stable=True)
    norm = math.comb(n - 1, k)

    baseline_sorted = torch.zeros_like(sorted_rewards)
    for i_idx in range(n):
        rank_i = i_idx + 1
        if rank_i < k:
            total = 0.0
            for rank_j in range(k, n + 1):
                total = total + math.comb(rank_j - 2, k - 1) * sorted_rewards[
                    ..., rank_j - 1
                ]
            baseline_sorted[..., i_idx] = total / norm
            continue

        total = 0.0
        for rank_j in range(k, rank_i):
            total = total + math.comb(rank_j - 1, k - 1) * sorted_rewards[
                ..., rank_j - 1
            ]
        for rank_j in range(rank_i + 1, n + 1):
            total = total + math.comb(rank_j - 2, k - 1) * sorted_rewards[
                ..., rank_j - 1
            ]

        baseline_sorted[..., i_idx] = total / norm

    baselines = torch.zeros_like(baseline_sorted)
    baselines.scatter_(-1, sorted_indices, baseline_sorted)
    return baselines.squeeze(0) if squeeze else baselines


def rspo_subloo_weights_reference(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Reference implementation of SubLOO closed form (Task 1.3 Prop 2.1).

    Args:
        rewards: Tensor of shape [n] or [batch, n].
        k: The K in Max@K (2 <= k <= n).

    Returns:
        Tensor of shape [n] or [batch, n] aligned to original order.
    """
    rewards_2d, squeeze = _as_2d(rewards)
    if rewards_2d.ndim != 2:
        raise ValueError("rewards must be 1D or 2D")

    n = rewards_2d.shape[-1]
    if k < 2 or k > n:
        raise ValueError("k out of range")

    sorted_rewards, sorted_indices = torch.sort(rewards_2d, dim=-1, stable=True)
    norm = math.comb(n, k)

    weights_sorted = torch.zeros_like(sorted_rewards)
    for i_idx in range(n):
        rank_i = i_idx + 1
        if rank_i < k:
            continue

        total = 0.0
        reward_i = sorted_rewards[..., i_idx]
        for m in range(k, rank_i + 1):
            coeff = math.comb(m - 2, k - 2)
            total = total + coeff * (reward_i - sorted_rewards[..., m - 2])

        weights_sorted[..., i_idx] = total / norm

    weights = torch.zeros_like(weights_sorted)
    weights.scatter_(-1, sorted_indices, weights_sorted)
    return weights.squeeze(0) if squeeze else weights


class TestPKPOAndRSPOFormulaValidation:
    """Validate estimator implementations against closed-form formulas."""

    @pytest.mark.parametrize("n", [3, 4, 6, 8])
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_pkpo_reward_estimator_closed_form(self, n: int, k: int) -> None:
        if k > n:
            pytest.skip("k must be <= n")
        torch.manual_seed(1000 + 10 * n + k)
        rewards = torch.randn(n, dtype=torch.float64)

        got = maxk_reward_estimate(rewards, k)
        expected = pkpo_maxgk_reward_estimate_reference(rewards, k)
        assert torch.allclose(got, expected, atol=1e-12)

    @pytest.mark.parametrize("n", [3, 4, 6, 8])
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_rspo_gradient_weights_closed_form(self, n: int, k: int) -> None:
        if k > n:
            pytest.skip("k must be <= n")
        torch.manual_seed(2000 + 10 * n + k)
        rewards = torch.randn(n, dtype=torch.float64)

        got = maxk_gradient_weights(rewards, k)
        expected = rspo_maxk_gradient_weights_reference(rewards, k)
        assert torch.allclose(got, expected, atol=1e-12)

    @pytest.mark.parametrize("n", [4, 5, 7])
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_pkpo_sample_loo_closed_form(self, n: int, k: int) -> None:
        if k >= n:
            pytest.skip("Sample-LOO requires k < n")
        torch.manual_seed(3000 + 10 * n + k)
        rewards = torch.randn(n, dtype=torch.float64)

        got = sample_loo_baseline(rewards, k)
        expected = pkpo_sample_loo_baseline_reference(rewards, k)
        assert torch.allclose(got, expected, atol=1e-12)

    @pytest.mark.parametrize("n", [3, 4, 6, 8])
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_rspo_subloo_closed_form(self, n: int, k: int) -> None:
        if k > n:
            pytest.skip("k must be <= n")
        torch.manual_seed(4000 + 10 * n + k)
        rewards = torch.randn(n, dtype=torch.float64)

        got = subloo_weights(rewards, k)
        expected = rspo_subloo_weights_reference(rewards, k)
        assert torch.allclose(got, expected, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 5, 8])
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_sum_s_equals_k_times_rho(self, n: int, k: int) -> None:
        if k > n:
            pytest.skip("k must be <= n")
        torch.manual_seed(5000 + 10 * n + k)
        rewards = torch.randn(n, dtype=torch.float64)

        s = maxk_gradient_weights(rewards, k)
        rho = maxk_reward_estimate(rewards, k)
        assert torch.allclose(s.sum(), torch.tensor(k, dtype=s.dtype) * rho, atol=1e-12)

