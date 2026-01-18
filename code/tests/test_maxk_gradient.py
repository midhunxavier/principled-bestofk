"""Unit tests for maxk_gradient module.

These tests verify correctness of the Max@K gradient weights by comparing
against exact enumeration over all k-subsets for small n.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from src.estimators.maxk_gradient import maxk_gradient_weights


def exact_gradient_weights(rewards: list[float], k: int) -> list[float]:
    """Compute exact gradient score weights via subset enumeration.

    For each sample i:
        s_i = (1 / C(n,k)) * sum_{S: |S|=k, i in S} max_{j in S} R_j

    Args:
        rewards: List of n reward values.
        k: Subset size.

    Returns:
        List of length n containing exact score weights.
    """
    n = len(rewards)
    assert 1 <= k <= n, f"Invalid k={k} for n={n}"

    totals = [0.0 for _ in range(n)]
    count = math.comb(n, k)
    for subset in itertools.combinations(range(n), k):
        subset_rewards = [rewards[i] for i in subset]
        subset_max = max(subset_rewards)
        for idx in subset:
            totals[idx] += subset_max

    return [total / count for total in totals]


class TestMaxKGradientWeights:
    """Tests for maxk_gradient_weights function."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_matches_enumeration(self, n: int, k: int) -> None:
        """Weights should match exact enumeration for small n."""
        if k > n:
            pytest.skip(f"k={k} > n={n}")

        torch.manual_seed(100 + n * 10 + k)
        rewards_list = torch.randn(n).tolist()

        weights = maxk_gradient_weights(torch.tensor(rewards_list), k)
        expected = exact_gradient_weights(rewards_list, k)

        assert torch.allclose(
            weights,
            torch.tensor(expected, dtype=weights.dtype),
            atol=1e-10,
        )

    def test_k1_equals_reward_over_n(self) -> None:
        """For k=1, s_i should equal R_i / n."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        weights = maxk_gradient_weights(rewards, k=1)
        expected = rewards / rewards.numel()
        assert torch.allclose(weights, expected)

    def test_k_equals_n_all_max(self) -> None:
        """For k=n, all weights should equal max reward."""
        rewards = torch.tensor([1.0, 5.0, 2.0, 4.0], dtype=torch.float64)
        weights = maxk_gradient_weights(rewards, k=4)
        expected = torch.full_like(rewards, rewards.max())
        assert torch.allclose(weights, expected)

    def test_batched_input(self) -> None:
        """Should handle 2D batched input."""
        torch.manual_seed(321)
        rewards = torch.randn(3, 5, dtype=torch.float64)
        weights = maxk_gradient_weights(rewards, k=2)

        assert weights.shape == rewards.shape
        for b in range(rewards.shape[0]):
            expected = exact_gradient_weights(rewards[b].tolist(), k=2)
            assert torch.allclose(
                weights[b], torch.tensor(expected, dtype=weights.dtype), atol=1e-10
            )

    def test_invalid_k_raises(self) -> None:
        """Should raise ValueError for invalid k."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="k must satisfy"):
            maxk_gradient_weights(rewards, k=0)
        with pytest.raises(ValueError, match="k must satisfy"):
            maxk_gradient_weights(rewards, k=4)

    def test_invalid_ndim_raises(self) -> None:
        """Should raise ValueError for 3D+ input."""
        rewards = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            maxk_gradient_weights(rewards, k=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
