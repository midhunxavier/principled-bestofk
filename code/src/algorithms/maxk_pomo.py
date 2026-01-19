"""RL4CO-compatible Max@K policy gradient (POMO-style).

This module implements Phase 3 Task 3.1 from `docs/PRD.md`:

    Create MaxK policy gradient class (RL4CO-compatible module).

The implementation follows the unbiased Max@K gradient score-weights from
Task 1.2 / Task 2.2 and optional variance reduction baselines from Task 1.3 /
Task 2.3.

Key references:
    - `docs/Tasks/Task1/task1.2/unbiasedness_proof.md` (Proposition 5.1)
    - `docs/Tasks/Task1/task1.3/loo_variance_reduction.md`
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.pomo import POMO

from src.estimators.baselines import apply_sample_loo, subloo_weights
from src.estimators.maxk_gradient import maxk_gradient_weights
from src.estimators.maxk_reward import maxk_reward_estimate

VarianceReduction = Literal["none", "sample_loo", "subloo"]


class MaxKPOMO(POMO):
    """POMO-style training with principled Max@K policy gradients.

    This class reuses RL4CO's POMO rollouts (multi-start decoding) but replaces
    the REINFORCE loss with an unbiased Max@K gradient estimator.

    Args:
        env: RL4CO environment.
        k: The K in Max@K (must satisfy 1 <= k <= num_starts at training time).
        variance_reduction: Variance reduction method to apply:
            - "none": no variance reduction
            - "sample_loo": subtract Sample-LOO baseline (requires n > k)
            - "subloo": use SubLOO hitchhiking-free weights (requires k >= 2)
        stable_sort: If True, use stable sorting for deterministic tie-breaking.
        policy: Optional policy module. If None, uses RL4CO's default policy for POMO.
        policy_kwargs: Keyword args for default RL4CO policy (ignored if policy is provided).
        baseline: Passed to RL4CO POMO constructor. POMO requires "shared", but the
            Max@K estimator does not use RL4CO's baseline in the loss.
        num_augment: Number of augmentations (validation/test only; training uses 0).
        augment_fn: Augmentation function name or callable (validation/test only).
        first_aug_identity: Whether to include identity augmentation first.
        feats: Features to augment.
        num_starts: Number of multi-start rollouts per instance (n). If None, uses
            environment default.
        **kwargs: Passed through to RL4CO POMO / Lightning module base classes.

    Raises:
        ValueError: If k < 1 or variance_reduction is invalid.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        *,
        k: int,
        variance_reduction: VarianceReduction = "none",
        stable_sort: bool = True,
        check_numerics: bool = False,
        debug_clamp_weights: float | None = None,
        policy: nn.Module | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: str | Callable = "dihedral8",
        first_aug_identity: bool = True,
        feats: list | None = None,
        num_starts: int | None = None,
        **kwargs: Any,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got k={k}")
        if variance_reduction not in ("none", "sample_loo", "subloo"):
            raise ValueError(
                "variance_reduction must be one of "
                f"('none', 'sample_loo', 'subloo'), got {variance_reduction!r}"
            )
        if debug_clamp_weights is not None and debug_clamp_weights <= 0:
            raise ValueError(
                "debug_clamp_weights must be > 0 when provided, got "
                f"debug_clamp_weights={debug_clamp_weights}"
            )

        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        super().__init__(
            env=env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            baseline=baseline,
            num_augment=num_augment,
            augment_fn=augment_fn,
            first_aug_identity=first_aug_identity,
            feats=feats,
            num_starts=num_starts,
            **kwargs,
        )

        self.k = int(k)
        self.variance_reduction: VarianceReduction = variance_reduction
        self.stable_sort = bool(stable_sort)
        self.check_numerics = bool(check_numerics)
        self.debug_clamp_weights = debug_clamp_weights

        self.save_hyperparameters(logger=False, ignore=["env", "policy"])

    def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Save hyperparameters while ignoring `env` and `policy`.

        RL4CO's upstream classes call `self.save_hyperparameters(...)` in their `__init__`.
        PyTorch Lightning warns when `nn.Module` objects (like `env`/`policy`) are stored as
        hyperparameters since they are already checkpointed via `state_dict`.
        """
        ignore = kwargs.get("ignore", [])
        if ignore is None:
            ignore = []
        if isinstance(ignore, (list, tuple, set)):
            ignore_set = set(ignore)
        else:
            ignore_set = {ignore}
        ignore_set.update({"env", "policy"})
        kwargs["ignore"] = list(ignore_set)

        if kwargs.get("frame") is None:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                kwargs["frame"] = frame.f_back

        super().save_hyperparameters(*args, **kwargs)

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict[str, Any],
        reward: torch.Tensor | None = None,
        log_likelihood: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Calculate Max@K loss for a training batch.

        Args:
            td: Current environment state tensordict (unused; kept for RL4CO API).
            batch: Input batch tensordict (unused; kept for RL4CO API).
            policy_out: Output dict from the policy forward pass (updated in-place).
            reward: Optional reward tensor of shape [batch, n]. If None, uses
                `policy_out["reward"]` (expected to be [batch * n], so passing the
                unbatchified reward is recommended).
            log_likelihood: Optional log-likelihood tensor of shape [batch, n]. If None,
                uses `policy_out["log_likelihood"]`.

        Returns:
            Updated `policy_out` containing `loss` and additional diagnostic keys.

        Raises:
            ValueError: If shapes are invalid or constraints for k / variance reduction fail.
        """
        rewards = reward if reward is not None else policy_out["reward"]
        logp = (
            log_likelihood
            if log_likelihood is not None
            else policy_out["log_likelihood"]
        )

        if rewards.ndim != 2:
            raise ValueError(
                "MaxKPOMO expects unbatchified rewards of shape [batch, n], "
                f"got rewards.shape={tuple(rewards.shape)}"
            )
        if logp.ndim != 2:
            raise ValueError(
                "MaxKPOMO expects unbatchified log_likelihood of shape [batch, n], "
                f"got log_likelihood.shape={tuple(logp.shape)}"
            )
        if rewards.shape != logp.shape:
            raise ValueError(
                "rewards and log_likelihood must have the same shape, "
                f"got rewards.shape={tuple(rewards.shape)}, log_likelihood.shape={tuple(logp.shape)}"
            )
        if self.check_numerics:
            if not torch.isfinite(rewards).all().item():
                num_bad = (~torch.isfinite(rewards)).sum().item()
                raise ValueError(f"Non-finite rewards detected (count={num_bad}).")
            if not torch.isfinite(logp).all().item():
                num_bad = (~torch.isfinite(logp)).sum().item()
                raise ValueError(
                    f"Non-finite log_likelihood detected (count={num_bad})."
                )

        n = rewards.shape[-1]
        k = self.k
        if k < 1 or k > n:
            raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")

        with torch.no_grad():
            if self.variance_reduction == "subloo":
                weights = subloo_weights(rewards, k, stable_sort=self.stable_sort)
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

        weights = self.advantage_scaler(weights)
        if self.debug_clamp_weights is not None:
            weights = weights.clamp(-self.debug_clamp_weights, self.debug_clamp_weights)

        if self.check_numerics:
            if not torch.isfinite(weights).all().item():
                num_bad = (~torch.isfinite(weights)).sum().item()
                raise ValueError(
                    f"Non-finite weights after scaling/clamping detected (count={num_bad})."
                )

        loss = -(weights.detach() * logp).sum(dim=-1).mean()

        policy_out.update({"loss": loss, "maxk_loss": loss, "rho_hat": rho_hat})
        if self.check_numerics:
            policy_out.update(
                {
                    "weights_mean": weights.mean(),
                    "weights_std": weights.std(unbiased=False),
                    "weights_abs_max": weights.abs().max(),
                }
            )
        return policy_out
