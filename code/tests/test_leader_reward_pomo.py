"""Unit tests for LeaderRewardPOMO RL4CO-compatible module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from src.algorithms.leader_reward import LeaderRewardPOMO


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


class TestLeaderRewardPOMOLoss:
    @pytest.mark.filterwarnings(
        "ignore:Attribute 'policy' is an instance of `nn\\.Module`:UserWarning"
    )
    def test_alpha_zero_matches_pomo_loss(self) -> None:
        POMO = pytest.importorskip("rl4co.models.zoo.pomo").POMO  # type: ignore[attr-defined]

        torch.manual_seed(0)
        batch_size, n = 3, 5

        pomo = POMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
        )
        leader = LeaderRewardPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            alpha=0.0,
        )

        rewards = torch.randn(batch_size, n)
        logp = torch.randn(batch_size, n, requires_grad=True)

        out_pomo: dict = {}
        out_leader: dict = {}
        out_pomo = pomo.calculate_loss(
            _td(batch_size), _td(batch_size), out_pomo, rewards, logp
        )
        out_leader = leader.calculate_loss(
            _td(batch_size), _td(batch_size), out_leader, rewards, logp
        )

        assert torch.allclose(out_leader["loss"], out_pomo["loss"])

    def test_loss_matches_reference(self) -> None:
        torch.manual_seed(0)
        batch_size, n = 2, 6
        alpha = 0.7

        model = LeaderRewardPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            alpha=alpha,
        )

        rewards = torch.randn(batch_size, n)
        logp = torch.randn(batch_size, n, requires_grad=True)

        mean = rewards.mean(dim=-1, keepdim=True)
        base_adv = rewards - mean
        r_max, leader_idx = rewards.max(dim=-1, keepdim=True)
        mask = torch.zeros_like(rewards)
        mask.scatter_(-1, leader_idx, 1.0)
        beta = r_max - mean
        advantage = base_adv + alpha * mask * beta

        expected_loss = -(advantage.detach() * logp).mean()

        out: dict = {}
        out = model.calculate_loss(_td(batch_size), _td(batch_size), out, rewards, logp)
        assert torch.allclose(out["loss"], expected_loss)

    def test_requires_unbatchified_shapes(self) -> None:
        batch_size, n = 2, 4
        model = LeaderRewardPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            alpha=0.0,
        )

        rewards = torch.randn(n)  # wrong shape
        logp = torch.randn(batch_size, n)

        with pytest.raises(ValueError, match="expects unbatchified rewards"):
            model.calculate_loss(_td(batch_size), _td(batch_size), {}, rewards, logp)

    def test_check_numerics_raises_on_nonfinite_inputs(self) -> None:
        batch_size, n = 2, 4
        model = LeaderRewardPOMO(
            _DummyEnv(),
            policy=_DummyPolicy(),
            num_augment=1,
            num_starts=n,
            alpha=0.0,
            check_numerics=True,
        )

        rewards = torch.randn(batch_size, n)
        rewards[0, 0] = float("nan")
        logp = torch.randn(batch_size, n)

        with pytest.raises(ValueError, match="Non-finite rewards"):
            model.calculate_loss(_td(batch_size), _td(batch_size), {}, rewards, logp)
