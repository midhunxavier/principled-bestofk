"""RL4CO-compatible Leader Reward baseline (POMO-style).

Leader Reward (Wang et al., 2024) is a heuristic modification to POMO's shared
baseline advantage that adds an extra bonus to the best ("leader") trajectory
within the multi-start rollouts.

This implementation follows the formulation described in:
`docs/Tasks/Task1/task1.3/loo_variance_reduction.md` §5.1.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.pomo import POMO

LeaderBonus = Literal["max_minus_mean"]


class LeaderRewardPOMO(POMO):
    """POMO-style training with the Leader Reward heuristic.

    Leader Reward modifies the shared-baseline advantage as:

        A_i = (R_i - mean(R)) + alpha * 1[i == argmax_j R_j] * beta

    where beta is typically `(R_max - mean(R))`.

    Args:
        env: RL4CO environment.
        alpha: Leader bonus scaling. Setting alpha=0 reduces to standard POMO.
        bonus: Which leader bonus term to use. Currently only "max_minus_mean".
        policy: Optional policy module. If None, uses RL4CO's default policy for POMO.
        policy_kwargs: Keyword args for default RL4CO policy (ignored if policy is provided).
        baseline: Passed to RL4CO POMO constructor (must be "shared").
        num_augment: Number of augmentations (validation/test only; training uses 0).
        augment_fn: Augmentation function name or callable (validation/test only).
        first_aug_identity: Whether to include identity augmentation first.
        feats: Features to augment.
        num_starts: Number of multi-start rollouts per instance (n). If None, uses
            environment default.
        **kwargs: Passed through to RL4CO POMO / Lightning module base classes.

    Raises:
        ValueError: If alpha < 0 or bonus is invalid.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        *,
        alpha: float = 1.0,
        bonus: LeaderBonus = "max_minus_mean",
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
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got alpha={alpha}")
        if bonus not in ("max_minus_mean",):
            raise ValueError(f"bonus must be one of ('max_minus_mean',), got {bonus!r}")
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

        self.alpha = float(alpha)
        self.bonus: LeaderBonus = bonus
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
        """Calculate Leader Reward loss for a training batch.

        Args:
            td: Current environment state tensordict.
            batch: Input batch tensordict. Kept for RL4CO API compatibility.
            policy_out: Output dict from the policy forward pass (updated in-place).
            reward: Optional reward tensor of shape [batch, n]. If None, uses
                `policy_out["reward"]` (expected to be [batch * n]).
            log_likelihood: Optional log-likelihood tensor of shape [batch, n]. If None,
                uses `policy_out["log_likelihood"]`.

        Returns:
            Updated `policy_out` containing `loss` and additional diagnostic keys.

        Raises:
            ValueError: If shapes are invalid.
        """
        rewards = reward if reward is not None else policy_out["reward"]
        logp = (
            log_likelihood
            if log_likelihood is not None
            else policy_out["log_likelihood"]
        )

        if rewards.ndim != 2:
            raise ValueError(
                "LeaderRewardPOMO expects unbatchified rewards of shape [batch, n], "
                f"got rewards.shape={tuple(rewards.shape)}"
            )
        if logp.ndim != 2:
            raise ValueError(
                "LeaderRewardPOMO expects unbatchified log_likelihood of shape [batch, n], "
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

        # Base shared-baseline advantage: R_i - mean(R)
        with torch.no_grad():
            bl_val, bl_loss = self.baseline.eval(td, rewards, self.env)

            advantage = rewards - bl_val

            if self.alpha != 0.0:
                r_max, leader_idx = rewards.max(dim=-1, keepdim=True)
                leader_mask = torch.zeros_like(rewards)
                leader_mask.scatter_(-1, leader_idx, 1.0)

                if self.bonus == "max_minus_mean":
                    beta = r_max - bl_val
                else:  # pragma: no cover
                    raise RuntimeError(f"Unhandled bonus mode: {self.bonus}")

                advantage = advantage + (self.alpha * leader_mask * beta)

        if self.check_numerics:
            if not torch.isfinite(advantage).all().item():
                num_bad = (~torch.isfinite(advantage)).sum().item()
                raise ValueError(f"Non-finite advantage detected (count={num_bad}).")

        advantage = self.advantage_scaler(advantage)
        if self.debug_clamp_weights is not None:
            advantage = advantage.clamp(
                -self.debug_clamp_weights, self.debug_clamp_weights
            )
        if self.check_numerics:
            if not torch.isfinite(advantage).all().item():
                num_bad = (~torch.isfinite(advantage)).sum().item()
                raise ValueError(
                    f"Non-finite advantage after scaling/clamping detected (count={num_bad})."
                )

        reinforce_loss = -(advantage.detach() * logp).mean()
        loss = reinforce_loss + bl_loss

        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        if self.check_numerics:
            policy_out.update(
                {
                    "advantage_mean": advantage.mean(),
                    "advantage_std": advantage.std(unbiased=False),
                    "advantage_abs_max": advantage.abs().max(),
                }
            )
        return policy_out
