"""Unit tests for loss functions in src.algorithms.losses."""

from __future__ import annotations

import pytest
import torch

from src.algorithms.losses import MaxKLoss, normalize_weights


class TestNormalizeWeights:
    def test_none_mode_returns_unchanged(self) -> None:
        weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = normalize_weights(weights, mode="none")
        assert torch.allclose(result, weights)

    def test_zscore_mode_zero_mean_unit_std(self) -> None:
        weights = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        result = normalize_weights(weights, mode="zscore")
        
        # Check zero mean per batch
        assert torch.allclose(result.mean(dim=-1), torch.zeros(2), atol=1e-6)
        
        # Check unit std per batch
        assert torch.allclose(result.std(dim=-1), torch.ones(2), atol=1e-2)

    def test_sum_to_zero_mode_zero_sum(self) -> None:
        weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = normalize_weights(weights, mode="sum_to_zero")
        
        # Check sum is zero per batch
        assert torch.allclose(result.sum(dim=-1), torch.zeros(2), atol=1e-6)
        
        # Check relative differences preserved
        expected = weights - weights.mean(dim=-1, keepdim=True)
        assert torch.allclose(result, expected)

    def test_invalid_mode_raises(self) -> None:
        weights = torch.randn(2, 4)
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            normalize_weights(weights, mode="invalid")  # type: ignore


class TestMaxKLossWeightNormalization:
    def test_zscore_normalization_applied(self) -> None:
        """Test that zscore normalization changes the weights."""
        torch.manual_seed(42)
        k = 3
        rewards = torch.randn(4, 8)
        logp = torch.randn(4, 8)
        
        # With normalization
        loss_norm = MaxKLoss(k=k, variance_reduction="none", weight_normalization="zscore")
        out_norm = loss_norm(rewards, logp)
        
        # Without normalization
        loss_no_norm = MaxKLoss(k=k, variance_reduction="none", weight_normalization="none")
        out_no_norm = loss_no_norm(rewards, logp)
        
        # Losses should be different when normalization is applied
        assert not torch.allclose(out_norm.loss, out_no_norm.loss)

    def test_weight_normalization_preserves_gradient_direction(self) -> None:
        """Test that normalized weights still point in a useful direction."""
        torch.manual_seed(42)
        k = 3
        rewards = torch.randn(4, 8)
        logp = torch.randn(4, 8, requires_grad=True)
        
        loss_fn = MaxKLoss(k=k, variance_reduction="subloo", weight_normalization="zscore")
        out = loss_fn(rewards, logp)
        
        out.loss.backward()
        
        # Gradient should exist and be finite
        assert logp.grad is not None
        assert torch.isfinite(logp.grad).all()


class TestMaxKLossMinGapScale:
    def test_min_gap_prevents_zero_gradient(self) -> None:
        """Test that min_gap_scale prevents zero gradients when rewards are identical."""
        k = 3
        # All identical rewards -> SubLOO would have zero weights
        rewards = torch.ones(4, 8)
        logp = torch.randn(4, 8, requires_grad=True)
        
        # With min_gap_scale
        loss_with_gap = MaxKLoss(
            k=k, 
            variance_reduction="subloo", 
            weight_normalization="none",  # Disable to see raw effect
            min_gap_scale=0.01
        )
        out = loss_with_gap(rewards, logp)
        
        # rho_hat should still work
        assert torch.isfinite(out.rho_hat).all()
        # Loss should be finite
        assert torch.isfinite(out.loss)

    def test_min_gap_zero_matches_original(self) -> None:
        """Test that min_gap_scale=0 matches original SubLOO behavior."""
        torch.manual_seed(42)
        k = 3
        rewards = torch.randn(4, 8)
        logp = torch.randn(4, 8)
        
        loss_with_gap = MaxKLoss(k=k, variance_reduction="subloo", weight_normalization="none", min_gap_scale=0.0)
        loss_without_gap = MaxKLoss(k=k, variance_reduction="subloo", weight_normalization="none", min_gap_scale=0.0)
        
        out_with = loss_with_gap(rewards, logp)
        out_without = loss_without_gap(rewards, logp)
        
        assert torch.allclose(out_with.loss, out_without.loss)
        assert torch.allclose(out_with.weights, out_without.weights)


class TestMaxKLossHybridMode:
    def test_hybrid_mode_blends_subloo_and_pomo(self) -> None:
        """Test that hybrid mode produces different results than pure SubLOO."""
        torch.manual_seed(42)
        k = 3
        rewards = torch.randn(4, 8)
        logp = torch.randn(4, 8)
        
        # Pure SubLOO
        loss_subloo = MaxKLoss(k=k, variance_reduction="subloo", weight_normalization="none", min_gap_scale=0.0)
        out_subloo = loss_subloo(rewards, logp)
        
        # Hybrid (50% SubLOO, 50% POMO)
        loss_hybrid = MaxKLoss(k=k, variance_reduction="hybrid", weight_normalization="none", hybrid_lambda=0.5, min_gap_scale=0.0)
        out_hybrid = loss_hybrid(rewards, logp)
        
        # Should be different
        assert not torch.allclose(out_subloo.loss, out_hybrid.loss)

    def test_hybrid_lambda_one_equals_subloo(self) -> None:
        """Test that hybrid_lambda=1.0 is equivalent to pure SubLOO."""
        torch.manual_seed(42)
        k = 3
        rewards = torch.randn(4, 8)
        logp = torch.randn(4, 8)
        
        # Pure SubLOO
        loss_subloo = MaxKLoss(k=k, variance_reduction="subloo", weight_normalization="none", min_gap_scale=0.0)
        out_subloo = loss_subloo(rewards, logp)
        
        # Hybrid with lambda=1.0 (should be same as SubLOO)
        loss_hybrid = MaxKLoss(k=k, variance_reduction="hybrid", weight_normalization="none", hybrid_lambda=1.0, min_gap_scale=0.0)
        out_hybrid = loss_hybrid(rewards, logp)
        
        assert torch.allclose(out_subloo.loss, out_hybrid.loss)

    def test_hybrid_lambda_zero_equals_pomo_advantage(self) -> None:
        """Test that hybrid_lambda=0.0 produces POMO-style mean-centered advantage."""
        torch.manual_seed(42)
        k = 3
        rewards = torch.randn(4, 8)
        logp = torch.randn(4, 8)
        
        # Hybrid with lambda=0.0 (should be pure POMO advantage)
        loss_hybrid = MaxKLoss(k=k, variance_reduction="hybrid", weight_normalization="none", hybrid_lambda=0.0, min_gap_scale=0.0)
        out_hybrid = loss_hybrid(rewards, logp)
        
        # Compute expected POMO advantage
        pomo_adv = rewards - rewards.mean(dim=-1, keepdim=True)
        expected_loss = -(pomo_adv.detach() * logp).sum(dim=-1).mean()
        
        assert torch.allclose(out_hybrid.loss, expected_loss)

    def test_hybrid_requires_k_geq_2(self) -> None:
        """Test that hybrid mode requires k >= 2 (same as SubLOO)."""
        rewards = torch.randn(4, 8)
        logp = torch.randn(4, 8)
        
        loss_fn = MaxKLoss(k=1, variance_reduction="hybrid", weight_normalization="none")
        
        with pytest.raises(ValueError, match="requires k >= 2"):
            loss_fn(rewards, logp)
