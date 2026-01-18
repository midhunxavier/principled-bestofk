"""Unit tests for LOO baselines."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from src.estimators.baselines import (
    apply_sample_loo,
    sample_loo_baseline,
    subloo_weights,
)
from src.estimators.maxk_gradient import maxk_gradient_weights


def exact_sample_loo_baseline(rewards: list[float], k: int) -> list[float]:
    """Compute Sample-LOO baselines via subset enumeration."""
    n = len(rewards)
    assert 1 <= k < n, f"Invalid k={k} for n={n}"
    baselines: list[float] = []
    for i in range(n):
        remaining = [rewards[j] for j in range(n) if j != i]
        total = 0.0
        count = 0
        for subset in itertools.combinations(range(n - 1), k):
            subset_rewards = [remaining[idx] for idx in subset]
            total += max(subset_rewards)
            count += 1
        assert count == math.comb(n - 1, k)
        baselines.append(total / count)
    return baselines


def exact_subloo_weights(rewards: list[float], k: int) -> list[float]:
    """Compute SubLOO weights via subset enumeration."""
    n = len(rewards)
    assert 2 <= k <= n, f"Invalid k={k} for n={n}"
    totals = [0.0 for _ in range(n)]
    count = math.comb(n, k)
    for subset in itertools.combinations(range(n), k):
        subset_rewards = [rewards[i] for i in subset]
        subset_max = max(subset_rewards)
        for idx in subset:
            other_rewards = [rewards[j] for j in subset if j != idx]
            second_max = max(other_rewards)
            totals[idx] += subset_max - second_max
    return [total / count for total in totals]


class TestSampleLOO:
    """Tests for Sample-LOO baseline computations."""

    @pytest.mark.parametrize("n", [4, 5, 6])
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_matches_enumeration(self, n: int, k: int) -> None:
        """Sample-LOO baselines should match enumeration."""
        if k >= n:
            pytest.skip("Sample-LOO requires k < n.")

        torch.manual_seed(123 + n * 10 + k)
        rewards_list = torch.randn(n).tolist()

        baselines = sample_loo_baseline(torch.tensor(rewards_list), k)
        expected = exact_sample_loo_baseline(rewards_list, k)

        assert torch.allclose(
            baselines,
            torch.tensor(expected, dtype=baselines.dtype),
            atol=1e-10,
        )

    def test_apply_sample_loo(self) -> None:
        """apply_sample_loo should subtract the baseline."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        s_weights = maxk_gradient_weights(rewards, k=2)
        baselines = sample_loo_baseline(rewards, k=2)
        adjusted = apply_sample_loo(s_weights, rewards, k=2)
        assert torch.allclose(adjusted, s_weights - baselines)

    def test_batched_input(self) -> None:
        """Should support batched Sample-LOO baselines."""
        torch.manual_seed(321)
        rewards = torch.randn(2, 5, dtype=torch.float64)
        baselines = sample_loo_baseline(rewards, k=2)
        assert baselines.shape == rewards.shape
        for b in range(rewards.shape[0]):
            expected = exact_sample_loo_baseline(rewards[b].tolist(), k=2)
            assert torch.allclose(
                baselines[b], torch.tensor(expected, dtype=baselines.dtype), atol=1e-10
            )

    def test_invalid_k_raises(self) -> None:
        """Should raise ValueError for invalid k."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="k must satisfy"):
            sample_loo_baseline(rewards, k=0)
        with pytest.raises(ValueError, match="k must satisfy"):
            sample_loo_baseline(rewards, k=3)

    def test_shape_mismatch_raises(self) -> None:
        """apply_sample_loo should require matching shapes."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        s_weights = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="must have the same shape"):
            apply_sample_loo(s_weights, rewards, k=1)

    def test_invalid_ndim_raises(self) -> None:
        """Should raise ValueError for 3D+ input."""
        rewards = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            sample_loo_baseline(rewards, k=1)

    def test_non_float_inputs_upcast(self) -> None:
        """Non-floating rewards should upcast to float64."""
        rewards = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        baselines = sample_loo_baseline(rewards, k=1)
        assert baselines.dtype == torch.float64


class TestSubLOO:
    """Tests for SubLOO weights."""

    @pytest.mark.parametrize("n", [4, 5, 6])
    @pytest.mark.parametrize("k", [2, 3])
    def test_matches_enumeration(self, n: int, k: int) -> None:
        """SubLOO weights should match enumeration."""
        if k > n:
            pytest.skip("Invalid k for given n.")

        torch.manual_seed(456 + n * 10 + k)
        rewards_list = torch.randn(n).tolist()

        weights = subloo_weights(torch.tensor(rewards_list), k)
        expected = exact_subloo_weights(rewards_list, k)

        assert torch.allclose(
            weights,
            torch.tensor(expected, dtype=weights.dtype),
            atol=1e-10,
        )

    def test_batched_input(self) -> None:
        """Should support batched SubLOO weights."""
        torch.manual_seed(654)
        rewards = torch.randn(2, 5, dtype=torch.float64)
        weights = subloo_weights(rewards, k=2)
        assert weights.shape == rewards.shape
        for b in range(rewards.shape[0]):
            expected = exact_subloo_weights(rewards[b].tolist(), k=2)
            assert torch.allclose(
                weights[b], torch.tensor(expected, dtype=weights.dtype), atol=1e-10
            )

    def test_invalid_k_raises(self) -> None:
        """Should raise ValueError for invalid k."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="k must satisfy"):
            subloo_weights(rewards, k=1)
        with pytest.raises(ValueError, match="k must satisfy"):
            subloo_weights(rewards, k=4)

    def test_invalid_ndim_raises(self) -> None:
        """Should raise ValueError for 3D+ input."""
        rewards = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            subloo_weights(rewards, k=2)

    def test_non_float_inputs_upcast(self) -> None:
        """Non-floating rewards should upcast to float64."""
        rewards = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        weights = subloo_weights(rewards, k=2)
        assert weights.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
