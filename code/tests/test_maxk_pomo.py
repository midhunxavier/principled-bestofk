"""Unit tests for MaxKPOMO RL4CO-compatible module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from src.algorithms.maxk_pomo import MaxKPOMO
from src.estimators.baselines import apply_sample_loo, subloo_weights
from src.estimators.maxk_gradient import maxk_gradient_weights


@dataclass(frozen=True)
class _DummyEnv:
    name: str = "dummy"


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_decode_type = "sampling"
        self.val_decode_type = "greedy"
        self.test_decode_type = "greedy"
        self._dummy_param = nn.Parameter(torch.zeros(()))


def _td(batch_size: int) -> TensorDict:
    return TensorDict({}, batch_size=[batch_size])


class TestMaxKPOMOLoss:
    def test_loss_matches_reference_no_variance_reduction(self) -> None:
        torch.manual_seed(0)
        batch_size, n, k = 3, 4, 2

        model = MaxKPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            k=k,
            variance_reduction="none",
            stable_sort=True,
        )

        rewards = torch.randn(batch_size, n, requires_grad=True)
        logp = torch.randn(batch_size, n, requires_grad=True)

        expected_weights = maxk_gradient_weights(rewards, k, stable_sort=True)
        expected_loss = -(expected_weights.detach() * logp).sum(dim=-1).mean()

        out: dict = {}
        out = model.calculate_loss(_td(batch_size), _td(batch_size), out, rewards, logp)
        assert torch.allclose(out["loss"], expected_loss)

        out["loss"].backward()
        assert rewards.grad is None
        assert logp.grad is not None
        assert torch.allclose(logp.grad, -expected_weights.detach() / batch_size)

    def test_loss_matches_reference_sample_loo(self) -> None:
        torch.manual_seed(0)
        batch_size, n, k = 2, 5, 2

        model = MaxKPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            k=k,
            variance_reduction="sample_loo",
            stable_sort=True,
        )

        rewards = torch.randn(batch_size, n, requires_grad=True)
        logp = torch.randn(batch_size, n, requires_grad=True)

        s = maxk_gradient_weights(rewards, k, stable_sort=True)
        expected_weights = apply_sample_loo(s, rewards, k, stable_sort=True)
        expected_loss = -(expected_weights.detach() * logp).sum(dim=-1).mean()

        out: dict = {}
        out = model.calculate_loss(_td(batch_size), _td(batch_size), out, rewards, logp)
        assert torch.allclose(out["loss"], expected_loss)

        out["loss"].backward()
        assert rewards.grad is None

    def test_loss_matches_reference_subloo(self) -> None:
        torch.manual_seed(0)
        batch_size, n, k = 2, 6, 3

        model = MaxKPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            k=k,
            variance_reduction="subloo",
            stable_sort=True,
        )

        rewards = torch.randn(batch_size, n, requires_grad=True)
        logp = torch.randn(batch_size, n, requires_grad=True)

        expected_weights = subloo_weights(rewards, k, stable_sort=True)
        expected_loss = -(expected_weights.detach() * logp).sum(dim=-1).mean()

        out: dict = {}
        out = model.calculate_loss(_td(batch_size), _td(batch_size), out, rewards, logp)
        assert torch.allclose(out["loss"], expected_loss)

        out["loss"].backward()
        assert rewards.grad is None

    def test_sample_loo_requires_n_gt_k(self) -> None:
        batch_size, n, k = 2, 4, 4
        model = MaxKPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            k=k,
            variance_reduction="sample_loo",
            stable_sort=True,
        )

        rewards = torch.randn(batch_size, n)
        logp = torch.randn(batch_size, n)

        with pytest.raises(ValueError, match="requires n > k"):
            model.calculate_loss(_td(batch_size), _td(batch_size), {}, rewards, logp)

    def test_requires_unbatchified_shapes(self) -> None:
        batch_size, n, k = 2, 4, 2
        model = MaxKPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            k=k,
            variance_reduction="none",
            stable_sort=True,
        )

        rewards = torch.randn(n)  # wrong shape
        logp = torch.randn(batch_size, n)

        with pytest.raises(ValueError, match="expects unbatchified rewards"):
            model.calculate_loss(_td(batch_size), _td(batch_size), {}, rewards, logp)


def test_can_instantiate_with_rl4co_env_smoke() -> None:
    """Smoke-test that the module can be constructed with a real RL4CO env."""
    get_env = pytest.importorskip("rl4co.envs").get_env  # type: ignore[attr-defined]

    env = get_env("tsp", generator_params={"num_loc": 5})
    model = MaxKPOMO(
        env,
        k=2,
        variance_reduction="none",
        num_augment=1,
        num_starts=5,
        policy_kwargs={"embed_dim": 16, "num_encoder_layers": 1, "num_heads": 1},
    )

    assert model.k == 2
    assert hasattr(model.policy, "train_decode_type")
    assert "multistart" in model.policy.train_decode_type
