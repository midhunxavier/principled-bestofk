"""Stability tests for large N values (overflow protection)."""

from __future__ import annotations

import pytest
import torch

from src.estimators.baselines import sample_loo_baseline, subloo_weights
from src.estimators.maxk_gradient import maxk_gradient_weights
from src.estimators.maxk_reward import maxk_reward_estimate


class TestStability:
    """Tests for numerical stability with large n."""

    @pytest.mark.parametrize("n", [64, 128])
    @pytest.mark.parametrize("k", [16, 32, 64])
    def test_large_n_stability(self, n: int, k: int) -> None:
        """Should produce finite outputs for large n where float32 would overflow."""
        if k > n:
            pytest.skip(f"k={k} > n={n}")

        torch.manual_seed(12345)
        # Use reasonably sized rewards that could cause overflow if multiplied by large coeffs
        # e.g., if coeff is 1e30 and reward is 1e2, product is 1e32 (fits in float32).
        # But if coeff is 1e37 and reward is 10, product is 1e38 (borderline/overflow).
        rewards = torch.randn(10, n, dtype=torch.float32) * 10.0

        # 1. Reward Estimate
        est = maxk_reward_estimate(rewards, k)
        assert torch.isfinite(est).all(), (
            f"Reward estimate infinite/NaN for n={n}, k={k}"
        )

        # 2. Gradient Weights
        grad_weights = maxk_gradient_weights(rewards, k)
        assert torch.isfinite(grad_weights).all(), (
            f"Gradient weights infinite/NaN for n={n}, k={k}"
        )

        # 3. Sample LOO
        if k < n:
            loo = sample_loo_baseline(rewards, k)
            assert torch.isfinite(loo).all(), (
                f"Sample LOO infinite/NaN for n={n}, k={k}"
            )

        # 4. SubLOO
        if k >= 2:
            sub = subloo_weights(rewards, k)
            assert torch.isfinite(sub).all(), f"SubLOO infinite/NaN for n={n}, k={k}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
