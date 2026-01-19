"""Smoke tests for the T3.2 training entrypoint."""

from __future__ import annotations

import pytest

from src.experiments import train_tsp


def test_parse_args_smoke(tmp_path) -> None:
    cfg = train_tsp.parse_args(
        [
            "--algorithm",
            "pomo",
            "--num_loc",
            "5",
            "--output_dir",
            str(tmp_path),
            "--max_epochs",
            "1",
        ]
    )
    assert cfg.algorithm == "pomo"
    assert cfg.num_loc == 5
    assert cfg.output_dir == tmp_path
    assert cfg.max_epochs == 1


def test_can_build_env_and_model_smoke(tmp_path) -> None:
    pytest.importorskip("rl4co")

    cfg = train_tsp.parse_args(
        [
            "--algorithm",
            "maxk_pomo",
            "--num_loc",
            "5",
            "--num_starts",
            "5",
            "--k",
            "2",
            "--output_dir",
            str(tmp_path),
            "--data_dir",
            str(tmp_path / "data"),
            "--train_data_size",
            "4",
            "--val_data_size",
            "4",
            "--test_data_size",
            "4",
        ]
    )

    env = train_tsp.make_env(cfg)
    model = train_tsp.make_model(cfg, env)

    assert hasattr(model, "calculate_loss")

