"""Gradient-level sanity checks on a tiny categorical toy problem.

This is a lightweight "principled" check: for a small action space, we can
compute the exact Max@K objective and its gradient by enumeration, and verify
that the Max@K policy-gradient surrogate is unbiased.
"""

from __future__ import annotations

import pytest
import torch

from src.algorithms.losses import MaxKLoss


def _exact_maxk_objective_and_grad(
    theta: torch.Tensor, action_rewards: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact J_maxK(theta) and ∇J by enumerating all k-tuples.

    Args:
        theta: Logits tensor of shape [A] with `requires_grad=True`.
        action_rewards: Reward per action of shape [A].
        k: Number of i.i.d. draws in Max@K.

    Returns:
        Tuple `(objective, grad)` where `objective` is scalar and `grad` has shape [A].
    """
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1D, got theta.shape={tuple(theta.shape)}")
    if action_rewards.ndim != 1:
        raise ValueError(
            "action_rewards must be 1D, "
            f"got action_rewards.shape={tuple(action_rewards.shape)}"
        )
    if theta.shape != action_rewards.shape:
        raise ValueError(
            "theta and action_rewards must have the same shape, "
            f"got theta.shape={tuple(theta.shape)}, action_rewards.shape={tuple(action_rewards.shape)}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}")

    num_actions = theta.shape[0]
    probs = torch.softmax(theta, dim=-1)

    actions = torch.cartesian_prod(*([torch.arange(num_actions)] * k))  # [A^k, k]
    tuple_probs = probs[actions].prod(dim=-1)  # [A^k]
    tuple_rewards = action_rewards[actions]  # [A^k, k]
    tuple_max = tuple_rewards.max(dim=-1).values  # [A^k]

    objective = (tuple_probs * tuple_max).sum()
    (grad,) = torch.autograd.grad(objective, theta, create_graph=False)
    return objective.detach(), grad.detach()


@pytest.mark.parametrize("variance_reduction", ["none", "sample_loo", "subloo"])
def test_maxk_loss_gradient_matches_exact_toy(variance_reduction: str) -> None:
    torch.manual_seed(0)

    dtype = torch.float64
    action_rewards = torch.tensor([0.0, 0.7, 1.3, 2.1], dtype=dtype)
    theta = torch.tensor([0.2, -0.1, 0.05, 0.4], dtype=dtype, requires_grad=True)

    k = 2
    n = 4
    num_trials = 20_000

    _, exact_grad = _exact_maxk_objective_and_grad(theta, action_rewards, k=k)

    with torch.no_grad():
        probs = torch.softmax(theta, dim=-1)
        actions = torch.distributions.Categorical(probs=probs).sample((num_trials, n))

    probs = torch.softmax(theta, dim=-1)
    logp = probs.log()[actions]  # [num_trials, n]
    rewards = action_rewards[actions]  # [num_trials, n]

    loss_fn = MaxKLoss(
        k=k,
        variance_reduction=variance_reduction,
        weight_normalization="none",  # Disable normalization for exact gradient match
        min_gap_scale=0.0,  # Disable min gap for exact gradient match
        stable_sort=True,
    )
    out = loss_fn(rewards, logp)

    # The surrogate objective is -loss. Its gradient should match ∇J_maxK.
    (grad_hat,) = torch.autograd.grad(-out.loss, theta, create_graph=False)

    torch.testing.assert_close(
        grad_hat.detach(),
        exact_grad,
        rtol=5e-2,
        atol=2e-2,
    )
