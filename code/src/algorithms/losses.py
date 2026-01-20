"""Objective / loss functions for RL4CO-style policy gradients.

RL4CO algorithms typically implement a `calculate_loss(...)` method on a Lightning
module (e.g., REINFORCE, POMO). To keep objectives reusable across different
architectures (AM, POMO, etc.), this module factors the *objective math* into
standalone loss helpers that operate on tensors like:

  - `rewards`:        [batch, n]
  - `log_likelihood`: [batch, n]

These losses are designed for stop-gradient semantics through any reward-based
weights (e.g., Max@K score weights).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from src.estimators.baselines import apply_sample_loo, subloo_weights
from src.estimators.maxk_gradient import maxk_gradient_weights
from src.estimators.maxk_reward import maxk_reward_estimate

VarianceReduction = Literal["none", "sample_loo", "subloo", "hybrid"]
WeightNormalization = Literal["none", "zscore", "sum_to_zero"]
ScaleFn = Callable[[torch.Tensor], torch.Tensor]


def normalize_weights(
    weights: torch.Tensor,
    mode: WeightNormalization = "zscore",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize gradient weights to stabilize training.

    SubLOO and MaxK weights can have very different magnitudes compared to
    standard POMO advantages. This normalization ensures the effective learning
    rate is comparable.

    Args:
        weights: Tensor of shape [batch, n] with gradient weights.
        mode: Normalization mode:
            - "none": No normalization (return as-is).
            - "zscore": Zero-mean, unit-std normalization per batch.
            - "sum_to_zero": Subtract mean to ensure sum is zero (preserves scale).
        eps: Small constant for numerical stability in division.

    Returns:
        Normalized weights tensor of same shape.
    """
    if mode == "none":
        return weights
    elif mode == "zscore":
        w_mean = weights.mean(dim=-1, keepdim=True)
        w_std = weights.std(dim=-1, keepdim=True)
        return (weights - w_mean) / (w_std + eps)
    elif mode == "sum_to_zero":
        w_mean = weights.mean(dim=-1, keepdim=True)
        return weights - w_mean
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")


@dataclass(frozen=True)
class MaxKLossOutput:
    """Outputs for Max@K policy-gradient loss computation."""

    loss: torch.Tensor
    weights: torch.Tensor
    rho_hat: torch.Tensor


class MaxKLoss:
    """Compute Max@K policy-gradient loss from rewards and log-likelihoods.

    This implements the surrogate loss used by `MaxKPOMO`:

        L(θ) = - E[ sum_i w_i(R_1:n) * log π_θ(τ_i) ],

    with stop-gradient semantics through the weights `w_i`.

    Args:
        k: The K in Max@K (must satisfy 1 <= k <= n at call time).
        variance_reduction: Variance reduction method:
            - "none": no variance reduction
            - "sample_loo": subtract Sample-LOO baseline (requires n > k)
            - "subloo": use SubLOO hitchhiking-free weights (requires k >= 2)
            - "hybrid": blend SubLOO with POMO-style mean-centered advantage
        weight_normalization: How to normalize weights before computing loss:
            - "none": no normalization (original behavior)
            - "zscore": zero-mean, unit-std normalization (RECOMMENDED for SubLOO)
            - "sum_to_zero": subtract mean only (preserves scale)
        stable_sort: If True, use stable sorting for deterministic tie-breaking.
        check_numerics: If True, raise on NaN/inf in inputs or computed weights.
        debug_clamp_weights: Optional clamp of weights after scaling (biases the
            estimator; debug-only).
        min_gap_scale: Minimum gap as fraction of reward range for SubLOO. Prevents
            zero gradients when rewards are clustered. Set to 0 to disable.
        hybrid_lambda: Blending coefficient for hybrid mode. 1.0 = pure SubLOO,
            0.0 = pure POMO advantage. Recommended: 0.5-0.8.
    """

    def __init__(
        self,
        *,
        k: int,
        variance_reduction: VarianceReduction = "none",
        weight_normalization: WeightNormalization = "zscore",
        stable_sort: bool = True,
        check_numerics: bool = False,
        debug_clamp_weights: float | None = None,
        min_gap_scale: float = 0.01,
        hybrid_lambda: float = 0.5,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got k={k}")
        if variance_reduction not in ("none", "sample_loo", "subloo", "hybrid"):
            raise ValueError(
                "variance_reduction must be one of "
                f"('none', 'sample_loo', 'subloo', 'hybrid'), got {variance_reduction!r}"
            )
        if weight_normalization not in ("none", "zscore", "sum_to_zero"):
            raise ValueError(
                "weight_normalization must be one of "
                f"('none', 'zscore', 'sum_to_zero'), got {weight_normalization!r}"
            )
        if debug_clamp_weights is not None and debug_clamp_weights <= 0:
            raise ValueError(
                "debug_clamp_weights must be > 0 when provided, got "
                f"debug_clamp_weights={debug_clamp_weights}"
            )
        if min_gap_scale < 0:
            raise ValueError(
                f"min_gap_scale must be >= 0, got min_gap_scale={min_gap_scale}"
            )
        if not (0.0 <= hybrid_lambda <= 1.0):
            raise ValueError(
                f"hybrid_lambda must be in [0, 1], got hybrid_lambda={hybrid_lambda}"
            )

        self.k = int(k)
        self.variance_reduction: VarianceReduction = variance_reduction
        self.weight_normalization: WeightNormalization = weight_normalization
        self.stable_sort = bool(stable_sort)
        self.check_numerics = bool(check_numerics)
        self.debug_clamp_weights = debug_clamp_weights
        self.min_gap_scale = float(min_gap_scale)
        self.hybrid_lambda = float(hybrid_lambda)

    def compute_weights(
        self, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Max@K gradient score-weights (and rho_hat) from rewards.

        Args:
            rewards: Reward tensor of shape `[batch, n]`.

        Returns:
            Tuple `(weights, rho_hat)` where both are `[batch, n]` / `[batch]`
            aligned to the original sample order.

        Raises:
            ValueError: If shapes are invalid or k/variance reduction constraints fail.
        """
        if rewards.ndim != 2:
            raise ValueError(
                "rewards must have shape [batch, n], "
                f"got rewards.shape={tuple(rewards.shape)}"
            )

        n = rewards.shape[-1]
        k = self.k
        if k < 1 or k > n:
            raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")

        if self.check_numerics and not torch.isfinite(rewards).all().item():
            num_bad = (~torch.isfinite(rewards)).sum().item()
            raise ValueError(f"Non-finite rewards detected (count={num_bad}).")

        with torch.no_grad():
            if self.variance_reduction == "subloo":
                weights = subloo_weights(
                    rewards, k,
                    stable_sort=self.stable_sort,
                    min_gap_scale=self.min_gap_scale,
                )
            elif self.variance_reduction == "hybrid":
                # Hybrid mode: blend SubLOO weights with POMO-style mean-centered advantage
                # This provides gradient signal even when SubLOO gaps are small
                if k < 2:
                    raise ValueError(
                        "Hybrid variance reduction requires k >= 2, "
                        f"got k={k} (use variance_reduction='none' for k=1)"
                    )
                subloo_w = subloo_weights(
                    rewards, k,
                    stable_sort=self.stable_sort,
                    min_gap_scale=self.min_gap_scale,
                )
                # POMO-style advantage: center rewards around mean
                pomo_adv = rewards - rewards.mean(dim=-1, keepdim=True)
                # Blend: lambda * SubLOO + (1 - lambda) * POMO
                weights = (
                    self.hybrid_lambda * subloo_w
                    + (1.0 - self.hybrid_lambda) * pomo_adv
                )
            else:
                s = maxk_gradient_weights(rewards, k, stable_sort=self.stable_sort)
                if self.variance_reduction == "sample_loo":
                    if n <= k:
                        raise ValueError(
                            "Sample-LOO requires n > k, "
                            f"got n={n}, k={k} (set variance_reduction='none' or use SubLOO)"
                        )
                    weights = apply_sample_loo(
                        s, rewards, k, stable_sort=self.stable_sort
                    )
                else:
                    weights = s

            rho_hat = maxk_reward_estimate(rewards, k, stable_sort=self.stable_sort)

        if self.check_numerics:
            if not torch.isfinite(weights).all().item():
                num_bad = (~torch.isfinite(weights)).sum().item()
                raise ValueError(
                    f"Non-finite Max@K weights detected (count={num_bad})."
                )
            if not torch.isfinite(rho_hat).all().item():
                num_bad = (~torch.isfinite(rho_hat)).sum().item()
                raise ValueError(f"Non-finite rho_hat detected (count={num_bad}).")

        return weights, rho_hat

    def __call__(
        self,
        rewards: torch.Tensor,
        log_likelihood: torch.Tensor,
        *,
        scale_fn: ScaleFn | None = None,
    ) -> MaxKLossOutput:
        """Compute Max@K surrogate loss for policy gradients.

        Args:
            rewards: Reward tensor of shape `[batch, n]`.
            log_likelihood: Log-likelihood tensor of shape `[batch, n]`.
            scale_fn: Optional scaling/normalization applied to weights *before*
                the loss (e.g., RL4CO `RewardScaler`). This can be stateful.

        Returns:
            `MaxKLossOutput` containing `(loss, weights, rho_hat)`.

        Raises:
            ValueError: If shapes are invalid or numeric checks fail.
        """
        if rewards.ndim != 2:
            raise ValueError(
                "rewards must have shape [batch, n], "
                f"got rewards.shape={tuple(rewards.shape)}"
            )
        if log_likelihood.ndim != 2:
            raise ValueError(
                "log_likelihood must have shape [batch, n], "
                f"got log_likelihood.shape={tuple(log_likelihood.shape)}"
            )
        if rewards.shape != log_likelihood.shape:
            raise ValueError(
                "rewards and log_likelihood must have the same shape, "
                f"got rewards.shape={tuple(rewards.shape)}, log_likelihood.shape={tuple(log_likelihood.shape)}"
            )

        if self.check_numerics and not torch.isfinite(log_likelihood).all().item():
            num_bad = (~torch.isfinite(log_likelihood)).sum().item()
            raise ValueError(f"Non-finite log_likelihood detected (count={num_bad}).")

        weights, rho_hat = self.compute_weights(rewards)
        
        # Apply weight normalization to stabilize training
        # This is critical for SubLOO which has different magnitude than POMO advantages
        weights = normalize_weights(weights, mode=self.weight_normalization)
        
        if scale_fn is not None:
            weights = scale_fn(weights)

        if self.debug_clamp_weights is not None:
            weights = weights.clamp(-self.debug_clamp_weights, self.debug_clamp_weights)

        if self.check_numerics and not torch.isfinite(weights).all().item():
            num_bad = (~torch.isfinite(weights)).sum().item()
            raise ValueError(
                f"Non-finite weights after scaling/clamping detected (count={num_bad})."
            )

        loss = -(weights.detach() * log_likelihood).sum(dim=-1).mean()
        return MaxKLossOutput(loss=loss, weights=weights, rho_hat=rho_hat)


def effective_sample_size(weights: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Compute a simple ESS-style concentration metric from per-sample weights.

    This uses the common importance-weight ESS formula:

        ESS = (sum_i w_i)^2 / sum_i w_i^2

    Note: `w_i` here may be signed (policy-gradient weights). This metric is used
    as a *diagnostic* for weight concentration/cancellation, not as a strict IS ESS.

    Args:
        weights: Tensor of shape `[batch, n]`.
        eps: Small epsilon to avoid division by zero.

    Returns:
        Tensor of shape `[batch]` with ESS values in `[0, n]` (up to numerical error).
    """
    if weights.ndim != 2:
        raise ValueError(
            "weights must have shape [batch, n], "
            f"got weights.shape={tuple(weights.shape)}"
        )
    sum_w = weights.sum(dim=-1)
    sum_w2 = weights.square().sum(dim=-1)
    return sum_w.square() / (sum_w2 + eps)
