"""Unit tests for maxk_estimator module.

These tests verify correctness of the Max@K reward estimator by comparing
against exact enumeration over all k-subsets for small n.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from src.estimators.maxk_reward import maxk_reward_estimate, maxk_reward_weights


def exact_maxk_reward_enumeration(rewards: list[float], k: int) -> float:
    """Compute exact Max@K reward via subset enumeration.

    This is the reference implementation:
        (1 / C(n,k)) * sum_{S: |S|=k} max_{i in S} R_i

    Args:
        rewards: List of n reward values.
        k: Subset size.

    Returns:
        Exact expected max reward over k-subsets.
    """
    n = len(rewards)
    assert 1 <= k <= n, f"Invalid k={k} for n={n}"

    total = 0.0
    count = 0
    for subset in itertools.combinations(range(n), k):
        subset_rewards = [rewards[i] for i in subset]
        total += max(subset_rewards)
        count += 1

    assert count == math.comb(n, k)
    return total / count


class TestMaxKRewardWeights:
    """Tests for maxk_reward_weights function."""

    def test_weights_sum_to_one(self) -> None:
        """Weights should sum to 1 for any valid n, k."""
        for n in range(1, 10):
            for k in range(1, n + 1):
                weights = maxk_reward_weights(n, k)
                assert weights.shape == (n,)
                assert torch.isclose(
                    weights.sum(), torch.tensor(1.0, dtype=torch.float64)
                )

    def test_k1_uniform_weights(self) -> None:
        """For k=1, weights should be uniform 1/n."""
        for n in range(1, 10):
            weights = maxk_reward_weights(n, k=1)
            expected = torch.full((n,), 1.0 / n, dtype=torch.float64)
            assert torch.allclose(weights, expected)

    def test_k_equals_n_only_max(self) -> None:
        """For k=n, only the last (max) position should have weight 1."""
        for n in range(1, 10):
            weights = maxk_reward_weights(n, k=n)
            expected = torch.zeros(n, dtype=torch.float64)
            expected[-1] = 1.0
            assert torch.allclose(weights, expected)

    def test_bottom_ranks_zero(self) -> None:
        """For k > 1, bottom k-1 ranks should have zero weight."""
        n = 8
        for k in range(2, n + 1):
            weights = maxk_reward_weights(n, k)
            # First k-1 weights (0-indexed: 0 to k-2) should be zero
            assert torch.allclose(
                weights[: k - 1], torch.zeros(k - 1, dtype=torch.float64)
            )

    def test_invalid_k_raises(self) -> None:
        """Should raise ValueError for invalid k."""
        with pytest.raises(ValueError, match="k must satisfy"):
            maxk_reward_weights(5, k=0)
        with pytest.raises(ValueError, match="k must satisfy"):
            maxk_reward_weights(5, k=6)
        with pytest.raises(ValueError, match="k must satisfy"):
            maxk_reward_weights(5, k=-1)

    def test_device_and_dtype(self) -> None:
        """Should respect device and dtype arguments."""
        weights = maxk_reward_weights(5, 2, dtype=torch.float32)
        assert weights.dtype == torch.float32

        # CPU device (always available)
        weights = maxk_reward_weights(5, 2, device=torch.device("cpu"))
        assert weights.device == torch.device("cpu")

    def test_invalid_dtype_raises(self) -> None:
        """Should raise ValueError for non-floating dtype."""
        with pytest.raises(ValueError, match="dtype must be floating"):
            maxk_reward_weights(5, 2, dtype=torch.int64)


class TestMaxKRewardEstimate:
    """Tests for maxk_reward_estimate function."""

    @pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_matches_enumeration(self, n: int, k: int) -> None:
        """Estimator should match exact enumeration for small n."""
        if k > n:
            pytest.skip(f"k={k} > n={n}")

        # Use deterministic rewards for reproducibility
        torch.manual_seed(42 + n * 10 + k)
        rewards_list = torch.randn(n).tolist()

        # Our estimator
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float64)
        estimate = maxk_reward_estimate(rewards_tensor, k)

        # Exact enumeration
        expected = exact_maxk_reward_enumeration(rewards_list, k)

        assert torch.isclose(
            estimate, torch.tensor(expected, dtype=torch.float64), atol=1e-10
        ), f"n={n}, k={k}: got {estimate.item()}, expected {expected}"

    def test_k1_equals_mean(self) -> None:
        """For k=1, Max@1 estimate equals mean (risk-neutral)."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        estimate = maxk_reward_estimate(rewards, k=1)
        expected = rewards.mean()
        assert torch.isclose(estimate, expected)

    def test_k_equals_n_equals_max(self) -> None:
        """For k=n, Max@n estimate equals max."""
        rewards = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0], dtype=torch.float64)
        estimate = maxk_reward_estimate(rewards, k=5)
        expected = rewards.max()
        assert torch.isclose(estimate, expected)

    def test_1d_input(self) -> None:
        """Should handle 1D input and return scalar."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        estimate = maxk_reward_estimate(rewards, k=2)
        assert estimate.ndim == 0  # scalar

    def test_2d_input_batched(self) -> None:
        """Should handle 2D batched input."""
        batch_size = 3
        n = 5
        k = 2

        torch.manual_seed(123)
        rewards = torch.randn(batch_size, n, dtype=torch.float64)

        estimate = maxk_reward_estimate(rewards, k)
        assert estimate.shape == (batch_size,)

        # Verify each batch element matches enumeration
        for b in range(batch_size):
            expected = exact_maxk_reward_enumeration(rewards[b].tolist(), k)
            assert torch.isclose(
                estimate[b], torch.tensor(expected, dtype=torch.float64), atol=1e-10
            )

    def test_ties_handled(self) -> None:
        """Should handle tied rewards without error."""
        # All same rewards
        rewards = torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
        estimate = maxk_reward_estimate(rewards, k=2)
        # With all same values, max of any subset is 3.0
        assert torch.isclose(estimate, torch.tensor(3.0, dtype=torch.float64))

        # Some ties
        rewards = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float64)
        estimate = maxk_reward_estimate(rewards, k=2)
        expected = exact_maxk_reward_enumeration(rewards.tolist(), k=2)
        assert torch.isclose(estimate, torch.tensor(expected, dtype=torch.float64))

    def test_invalid_k_raises(self) -> None:
        """Should raise ValueError for invalid k."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="k must satisfy"):
            maxk_reward_estimate(rewards, k=0)
        with pytest.raises(ValueError, match="k must satisfy"):
            maxk_reward_estimate(rewards, k=5)

    def test_invalid_ndim_raises(self) -> None:
        """Should raise ValueError for 3D+ input."""
        rewards = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            maxk_reward_estimate(rewards, k=2)

    def test_monotonic_in_k(self) -> None:
        """Estimate should be monotonically increasing in k."""
        torch.manual_seed(999)
        rewards = torch.randn(8, dtype=torch.float64)

        estimates = [maxk_reward_estimate(rewards, k=k).item() for k in range(1, 9)]

        for i in range(len(estimates) - 1):
            assert estimates[i] <= estimates[i + 1] + 1e-10, (
                f"Not monotonic: k={i + 1} -> {estimates[i]}, k={i + 2} -> {estimates[i + 1]}"
            )

    def test_non_float_inputs_upcast(self) -> None:
        """Non-floating rewards should upcast to float64."""
        rewards = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        estimate = maxk_reward_estimate(rewards, k=2)
        assert estimate.dtype == torch.float64


class TestSpecificExamples:
    """Tests with specific hand-computed examples."""

    def test_n4_k2_example(self) -> None:
        """Hand-verified example: n=4, k=2."""
        # Rewards: [1, 2, 3, 4] (already sorted)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        # C(4,2) = 6 subsets:
        # {1,2} -> max=2, {1,3} -> max=3, {1,4} -> max=4
        # {2,3} -> max=3, {2,4} -> max=4, {3,4} -> max=4
        # Sum = 2 + 3 + 4 + 3 + 4 + 4 = 20
        # Expected = 20 / 6 = 10/3
        expected = 20.0 / 6.0

        estimate = maxk_reward_estimate(rewards, k=2)
        assert torch.isclose(estimate, torch.tensor(expected, dtype=torch.float64))

    def test_weights_n4_k2(self) -> None:
        """Verify weight values for n=4, k=2."""
        # Weights for sorted rewards R_(1) <= R_(2) <= R_(3) <= R_(4)
        # w_1 = 0 (rank < k)
        # w_2 = C(1,1)/C(4,2) = 1/6
        # w_3 = C(2,1)/C(4,2) = 2/6 = 1/3
        # w_4 = C(3,1)/C(4,2) = 3/6 = 1/2
        weights = maxk_reward_weights(4, 2)

        expected = torch.tensor([0.0, 1 / 6, 2 / 6, 3 / 6], dtype=torch.float64)
        assert torch.allclose(weights, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
